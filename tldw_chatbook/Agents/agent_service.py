# tldw_chatbook/Agents/agent_service.py
"""Wires the pure agent loop to the real provider, tools, and run store.

The ONLY impure Agents module: provider calls (chat_api_call), the
permission gate, sub-agent spawning, and AgentRunsDB persistence.
Runs synchronously — callers put it on a worker thread (Plan B).
"""
from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone
from typing import Callable

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from .agent_models import (
    AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT, RUN_DONE, RUN_ERROR,
    SPAWN_TOOL_NAME, STEP_ERROR, AgentConfig, AgentStep, ModelTurn,
    RunOutcome, ToolCall, ToolResult, clamp_child_budget,
)
from .agent_runtime import LoopDeps, render_tool_protocol, run_agent_loop
from .tool_catalog import (
    FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA, SPAWN_TOOL_SCHEMA,
    ToolCatalogRegistry, initial_disclosure,
)

SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused sub-agent. Complete the task you are given and "
    "reply with a concise result. You cannot ask the user questions.")

TRUNCATION_NOTICE = "\n[truncated]"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _default_chat_call():
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    return chat_api_call


def _response_text(resp) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


class AgentService:
    """Run one agent turn (primary + any sub-agents) and persist it."""

    def __init__(self, db: AgentRunsDB, registry: ToolCatalogRegistry,
                 chat_call: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.db = db
        self.registry = registry
        self.chat_call = chat_call or _default_chat_call()
        self.clock = clock

    # -- internals -------------------------------------------------------

    def _make_call_model(self, config: AgentConfig, api_endpoint: str,
                         runtime_schemas: list):
        def call_model(messages: list[dict], active_schemas: tuple) -> ModelTurn:
            protocol = render_tool_protocol(
                runtime_schemas + list(active_schemas))
            system_content = config.system_prompt
            if protocol:
                system_content = f"{config.system_prompt}\n\n{protocol}"
            payload = [{"role": "system", "content": system_content}]
            payload.extend(messages)
            resp = self.chat_call(
                api_endpoint=api_endpoint, messages_payload=payload,
                streaming=False, model=config.model)
            return ModelTurn(text=_response_text(resp))
        return call_model

    def _make_invoke_tool(self, config: AgentConfig,
                          disclosed_names: set):
        def invoke_tool(call: ToolCall) -> ToolResult:
            if (call.name not in config.allowed_tools
                    or call.name not in disclosed_names):
                return ToolResult(
                    ok=False, error=f"Tool not permitted: {call.name}")
            return self.registry.invoke_by_name(call.name, call.args)
        return invoke_tool

    def _persist(self, run_id: str, outcome: RunOutcome) -> None:
        stamp = _now_iso()
        step_dicts = []
        for step in outcome.steps:
            record = dataclasses.asdict(step)
            record["created_at"] = record["created_at"] or stamp
            step_dicts.append(record)
        self.db.append_steps(run_id, step_dicts)
        self.db.set_status(run_id, outcome.status,
                           result=outcome.final_text or None)

    def _run_one(self, *, conversation_id: str, messages: list[dict],
                 config: AgentConfig, api_endpoint: str,
                 should_cancel: Callable[[], bool], agent_kind: str,
                 task: str | None, parent_run_id: str | None
                 ) -> tuple[str, RunOutcome]:
        run_id = self.db.create_run(
            conversation_id=conversation_id, agent_kind=agent_kind,
            task=task, parent_run_id=parent_run_id,
            budget=dataclasses.asdict(config.budget))
        started = self.clock()

        active, offer_find_load = initial_disclosure(
            self.registry, config.budget)
        # Q7(a): the initial active set must respect the allow-list too —
        # the permission gate is a backstop, not the only checkpoint. A
        # disallowed tool must never even be disclosed to the model.
        active = [schema for schema in active
                 if schema.name in config.allowed_tools]
        disclosed_names = {schema.name for schema in active}
        runtime_schemas = []
        if config.budget.max_subagents > 0:
            runtime_schemas.append(SPAWN_TOOL_SCHEMA)
        if offer_find_load:
            runtime_schemas.extend([FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA])

        def find_tools(query: str):
            # Q7(b): never surface a disallowed tool through find_tools,
            # even though it exists in the catalog.
            return [entry for entry in self.registry.find(query)
                   if entry.name in config.allowed_tools]

        def load_schemas(ids: list):
            schemas = []
            for tool_id in ids:
                try:
                    schema = self.registry.load_schema(str(tool_id))
                except KeyError:
                    continue
                # Q7(c): never disclose a tool outside the allow-list.
                if schema.name not in config.allowed_tools:
                    continue
                # G3: an id whose name is already disclosed must be
                # filtered out BEFORE the room slice below — otherwise a
                # redundant re-load of an already-active tool both eats a
                # room slot it doesn't need and (because the loop's own
                # `active` list already holds the schema) desyncs this
                # gate's disclosed_names from the loop's actual active-set
                # size, letting the loop append a duplicate. Filtering
                # first keeps the two lists in lockstep at the cost of a
                # generic "No valid tools found to load" message on
                # redundant re-loads — an acceptable trade-off for cap
                # integrity (see PR review decision).
                if schema.name in disclosed_names:
                    continue
                schemas.append(schema)
            # Mirror the loop's own room-slicing (agent_runtime.py's
            # load_tools branch) so the gate-disclosed set never grows past
            # what the loop actually admits into `active`. disclosed_names
            # starts equal to the initial active set and only ever gains
            # names here, so its size always matches len(active) — the same
            # room computation the loop performs independently.
            room = config.budget.max_active_tools - len(disclosed_names)
            accepted = schemas[:max(room, 0)]
            for schema in accepted:
                disclosed_names.add(schema.name)
            return accepted

        def spawn(spawn_task: str) -> ToolResult:
            remaining = config.budget.max_wall_seconds - (
                self.clock() - started)
            child_config = AgentConfig(
                model=config.model,
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                allowed_tools=tuple(
                    n for n in config.allowed_tools
                    if n != SPAWN_TOOL_NAME),
                budget=clamp_child_budget(config.budget, remaining))
            _child_id, child_outcome = self._run_one(
                conversation_id=conversation_id,
                messages=[{"role": "user", "content": spawn_task}],
                config=child_config, api_endpoint=api_endpoint,
                should_cancel=should_cancel,
                agent_kind=AGENT_KIND_SUBAGENT, task=spawn_task,
                parent_run_id=run_id)
            text = child_outcome.final_text
            cap = config.budget.max_subagent_result_chars
            if len(text) > cap:
                text = text[:cap] + TRUNCATION_NOTICE
            if child_outcome.status != RUN_DONE:
                return ToolResult(
                    ok=False,
                    error=f"sub-agent {child_outcome.status}: {text}")
            return ToolResult(ok=True, content=text)

        deps = LoopDeps(
            call_model=self._make_call_model(
                config, api_endpoint, runtime_schemas),
            invoke_tool=self._make_invoke_tool(config, disclosed_names),
            spawn=spawn,
            find_tools=find_tools,
            load_schemas=load_schemas,
            should_cancel=should_cancel,
            clock=self.clock,
        )
        try:
            outcome = run_agent_loop(config, messages, active, deps)
        except Exception as exc:  # noqa: BLE001 — a run never raises out
            outcome = RunOutcome(
                status=RUN_ERROR,
                steps=[AgentStep(index=0, kind=STEP_ERROR,
                                 summary=str(exc)[:500])])
        self._persist(run_id, outcome)
        return run_id, outcome

    # -- public ----------------------------------------------------------

    def run_turn(self, *, conversation_id: str, messages: list[dict],
                 config: AgentConfig, api_endpoint: str,
                 should_cancel: Callable[[], bool] = lambda: False,
                 supersede_run_id: str | None = None
                 ) -> tuple[str, RunOutcome]:
        """Run one primary-agent turn (and any sub-agents it spawns).

        Wires the pure ``run_agent_loop`` to the real provider, the tool
        catalog/permission gate, and ``AgentRunsDB`` persistence. Runs
        synchronously — callers put this on a worker thread.

        Args:
            conversation_id: The owning Console conversation's id; also
                used to scope sub-agent fan-out counting.
            messages: The initial message history (role/content dicts) to
                seed the loop with — typically the conversation transcript
                plus any staged/RAG context.
            config: The primary agent's model, system prompt, allow-list,
                and budget.
            api_endpoint: The provider endpoint identifier passed through
                to ``chat_api_call``.
            should_cancel: Polled at step and tool-call boundaries; once it
                returns ``True`` the whole run tree stops and persists as
                ``cancelled``.
            supersede_run_id: When set, marks that prior run (and its
                sub-agent tree) ``superseded`` before starting this run —
                used by retry/regenerate/continue.

        Returns:
            A ``(run_id, outcome)`` tuple: the new primary run's id and its
            terminal ``RunOutcome``. The run record (and any sub-agent run
            records) are persisted before this returns.
        """
        if supersede_run_id:
            self.db.supersede_run_tree(supersede_run_id)
        return self._run_one(
            conversation_id=conversation_id, messages=messages,
            config=config, api_endpoint=api_endpoint,
            should_cancel=should_cancel, agent_kind=AGENT_KIND_PRIMARY,
            task=None, parent_run_id=None)
