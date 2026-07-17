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
from typing import Callable, Protocol

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from .agent_models import (
    AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT, RUN_DONE, RUN_ERROR,
    SPAWN_TOOL_NAME, STEP_ERROR, AgentConfig, AgentStep, ModelTurn,
    RunOutcome, ToolCall, ToolResult, clamp_child_budget,
)
from .agent_runtime import LoopDeps, render_tool_protocol, run_agent_loop
from .native_tools import (
    ensure_tool_call_ids, parse_native_tool_calls,
    provider_supports_native_tools, schemas_to_openai_tools,
)
from .tool_catalog import (
    FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA, SPAWN_TOOL_SCHEMA,
    ToolCatalogRegistry, initial_disclosure,
)

SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused sub-agent. Complete the task you are given and "
    "reply with a concise result. You cannot ask the user questions.")

TRUNCATION_NOTICE = "\n[truncated]"


class SkillRunner(Protocol):
    """Executes a skill-tool call as a budget-counted, spawn-wired sub-agent.

    Implemented by ``console_agent_bridge._BridgeSkillRunner``; a plain fake
    in tests. ``run`` is handed THIS run's own ``spawn`` closure so a skill's
    rendered prompt executes exactly like any other sub-agent -- cancellable
    via ``should_cancel``, DB-lineage-tracked via ``parent_run_id``, and
    result-capped -- never a bespoke, unbounded execution path. This is the
    replacement for the pre-wiring path traced in Task 11, where a skill
    tool routed to ``SkillToolProvider.invoke`` (which raises by design and
    aborted the whole run).
    """

    def is_skill_tool(self, name: str) -> bool:
        """Return whether ``name`` is a skill tool this runner handles."""
        ...

    def run(self, name: str, args: str,
            spawn: Callable[..., "ToolResult"]) -> "ToolResult":
        """Render skill ``name`` with ``args`` and run it via ``spawn``.

        Args:
            name: The skill's tool name (as it appears in
                ``config.allowed_tools``).
            args: The raw ``args`` string the model passed (the tool
                schema's single ``args`` property -- see
                ``SkillToolProvider.load_schema``).
            spawn: This run's own spawn closure -- ``spawn(task, *,
                allowed_tools=None)`` -- so the rendered skill prompt runs
                as a normal budget-counted sub-agent of THIS run.

        Returns:
            The sub-agent's result, wrapped as a ``ToolResult`` exactly the
            way ``spawn`` itself returns one.
        """
        ...


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


def _response_message(resp) -> dict:
    try:
        message = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}
    return message if isinstance(message, dict) else {}


class AgentService:
    """Run one agent turn (primary + any sub-agents) and persist it."""

    def __init__(self, db: AgentRunsDB, registry: ToolCatalogRegistry,
                 chat_call: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 on_step: Callable[[AgentStep, str], None] | None = None,
                 skill_runner: SkillRunner | None = None,
                 ) -> None:
        self.db = db
        self.registry = registry
        self.chat_call = chat_call or _default_chat_call()
        self.clock = clock
        self._on_step = on_step
        self.skill_runner = skill_runner

    # -- internals -------------------------------------------------------

    def _make_call_model(self, config: AgentConfig, api_endpoint: str,
                         runtime_schemas: list):
        native = (config.native_tools
                  and provider_supports_native_tools(api_endpoint))
        # task-245: one render per active-set change, not per turn. Keyed by
        # schema NAMES (the set only ever grows via load_tools — AC #2), and
        # scoped to this closure = this run, so sub-agents (their own
        # _run_one -> their own closure) never share a cache. Byte-stable
        # repeated turns are the precondition for provider-side prompt
        # caching (see Docs/superpowers/reviews/
        # 2026-07-17-provider-prompt-caching-note.md).
        protocol_key: tuple | None = None
        protocol_text = ""

        def call_model(messages: list[dict], active_schemas: tuple) -> ModelTurn:
            nonlocal protocol_key, protocol_text
            schemas = runtime_schemas + list(active_schemas)
            system_content = config.system_prompt
            call_kwargs: dict = {}
            if native:
                # Native mode: the provider carries the tool catalog in
                # tools= — no fence-protocol section in the system prompt.
                tools = schemas_to_openai_tools(schemas)
                if tools:
                    call_kwargs["tools"] = tools
            else:
                key = tuple(s.name for s in schemas)
                if key != protocol_key:
                    protocol_text = render_tool_protocol(schemas)
                    protocol_key = key
                if protocol_text:
                    system_content = f"{config.system_prompt}\n\n{protocol_text}"
            payload = [{"role": "system", "content": system_content}]
            payload.extend(messages)
            resp = self.chat_call(
                api_endpoint=api_endpoint, messages_payload=payload,
                streaming=False, model=config.model, **call_kwargs)
            text = _response_text(resp)
            if not native:
                return ModelTurn(text=text)
            message = _response_message(resp)
            # Id-less entries get synthesized ids BEFORE parsing, and the
            # SAME normalized list feeds the assistant echo — the echo and
            # its role="tool" replies must always pair by id (PR #648
            # review: a split convention 400s on strict providers).
            raw_calls = ensure_tool_call_ids(message.get("tool_calls"))
            if raw_calls:
                message = {**message, "tool_calls": raw_calls}
            tool_calls = parse_native_tool_calls(message)
            assistant_message = None
            if tool_calls:
                assistant_message = {
                    "role": "assistant", "content": text,
                    "tool_calls": raw_calls}
            return ModelTurn(text=text, tool_calls=tool_calls,
                             assistant_message=assistant_message)
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
                    # task-244 AC#3: models often echo a bare tool NAME from
                    # a find_tools result line instead of the catalog id —
                    # resolve it before giving up on this entry, instead of
                    # burning the whole round on a generic load error.
                    resolved = self.registry.resolve_name(str(tool_id))
                    if resolved is None:
                        continue
                    try:
                        schema = self.registry.load_schema(resolved)
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
                # PR #655 review (Gemini): one batch can reach the SAME
                # schema twice — its bare name plus its catalog id, or a
                # repeated id. disclosed_names only guards against PRIOR
                # rounds (it is updated after this loop), so without an
                # in-batch dedupe both copies would append and desync the
                # loop's active list from this gate's disclosed set.
                if any(s.name == schema.name for s in schemas):
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

        sub_agent_spawns = 0

        def spawn(spawn_task: str, *,
                  allowed_tools: tuple[str, ...] | None = None) -> ToolResult:
            nonlocal sub_agent_spawns
            # Task-12 review Finding 2: this closure is THE single spawn
            # path -- the loop calls it directly for the native
            # spawn_subagent tool (agent_runtime.py), and invoke_tool's
            # skill branch below calls it via skill_runner.run. Gating and
            # incrementing the shared counter HERE, before any child run is
            # created, enforces one combined sub-agent ceiling across both
            # paths regardless of call order. (Previously each path checked
            # its own independent counter -- the loop's own `spawned` and
            # this service's now-removed `skill_spawns` -- so an operator
            # ceiling of 1 could permit 2 sub-agent runs.) The loop's own
            # counter stays untouched as a redundant secondary bound that
            # is never reached first.
            if sub_agent_spawns >= config.budget.max_subagents:
                return ToolResult(
                    ok=False, error="sub-agent budget exhausted")
            sub_agent_spawns += 1
            remaining = config.budget.max_wall_seconds - (
                self.clock() - started)
            # Q6/Task-12: an explicit override (a skill's own narrowed,
            # builtins-only allow-list -- see SkillRunner.run) replaces the
            # default entirely; the default itself preserves the shipped
            # behavior (spawn_subagent's child inherits the parent's
            # allow-list minus the spawn tool itself, so a depth-1 child
            # never re-offers spawn_subagent) -- MINUS any skill-tool names
            # too (pre-merge review MINOR 3). An ordinary native-spawn
            # child can never actually run a skill (max_subagents is
            # always clamped to 0 for every child, one-deep-only by
            # construction), so a skill name surviving into its allow-list
            # only meant a call to it fell through to that numeric
            # budget-exhausted refusal below instead of the permission
            # gate every other disallowed tool hits -- fragile (an
            # incidental side effect of the budget clamp, not a modeled
            # boundary) and inconsistent with the skill-driven child's own
            # explicit builtins-only allow-list. Excluding skill names
            # here too means a child can neither discover (find_tools/
            # disclosure) nor invoke one; a stray direct call still gets a
            # graceful "Tool not permitted" ToolResult from invoke_tool's
            # skill branch, never reaching skill_runner.run.
            child_allowed_tools = (
                allowed_tools if allowed_tools is not None
                else tuple(
                    n for n in config.allowed_tools
                    if n != SPAWN_TOOL_NAME
                    and not (self.skill_runner is not None
                            and self.skill_runner.is_skill_tool(n))
                ))
            child_config = AgentConfig(
                model=config.model,
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                allowed_tools=child_allowed_tools,
                budget=clamp_child_budget(config.budget, remaining),
                native_tools=config.native_tools)
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

        # Skill-aware invoke_tool, built AFTER spawn (it closes over it): a
        # skill-tool call never reaches the registry/ToolProvider.invoke
        # path (SkillToolProvider.invoke raises by design -- Task 11 traced
        # that pre-wiring path as a loud full-run abort). Instead it routes
        # through skill_runner.run, which renders the skill and calls THIS
        # run's spawn -- so it is budget-counted (via spawn's own shared
        # sub_agent_spawns counter -- see Finding 2 above), cancellable, and
        # DB-lineage-tracked exactly like a spawn_subagent call.
        builtin_invoke_tool = self._make_invoke_tool(config, disclosed_names)

        def invoke_tool(call: ToolCall) -> ToolResult:
            if (self.skill_runner is not None
                    and self.skill_runner.is_skill_tool(call.name)):
                # Task-12 review Finding 1: a skill tool must pass the SAME
                # two-part gate as an ordinary catalog tool (mirrors
                # _make_invoke_tool above) -- allowed_tools is the
                # permission boundary, but disclosed_names (seeded by
                # initial disclosure and grown only via load_tools, exactly
                # like a builtin) is the other half. Checking allowed_tools
                # alone let an undisclosed skill name execute the instant
                # the model guessed it, even behind a >8-tool catalog where
                # progressive disclosure is supposed to gate exactly this.
                if (call.name not in config.allowed_tools
                        or call.name not in disclosed_names):
                    return ToolResult(
                        ok=False, error=f"Tool not permitted: {call.name}")
                # Cheap early exit before rendering the skill: the
                # authoritative check-and-increment lives in `spawn` itself
                # (shared with the native spawn_subagent path), so the
                # combined ceiling holds regardless of call order even
                # without this line -- it only saves an unnecessary
                # render/trust round-trip once the shared budget is spent.
                if sub_agent_spawns >= config.budget.max_subagents:
                    return ToolResult(
                        ok=False, error="sub-agent budget exhausted")
                return self.skill_runner.run(
                    call.name, str(call.args.get("args", "")), spawn)
            return builtin_invoke_tool(call)

        deps = LoopDeps(
            call_model=self._make_call_model(
                config, api_endpoint, runtime_schemas),
            invoke_tool=invoke_tool,
            spawn=spawn,
            find_tools=find_tools,
            load_schemas=load_schemas,
            should_cancel=should_cancel,
            clock=self.clock,
            on_step=((lambda s: self._on_step(s, agent_kind))
                     if self._on_step is not None else (lambda s: None)),
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
        # Per-run scope for the registry's owner-map cache (tool_catalog's
        # _owner_and_id): reset here, once, at the top of the run tree —
        # covers the primary turn AND any sub-agents it spawns via
        # _run_one, since they never call run_turn themselves. The catalog
        # is listed fresh at this point, so skill CRUD since the last run
        # is always picked up with no separate invalidation signal needed.
        self.registry.reset_catalog_cache()
        return self._run_one(
            conversation_id=conversation_id, messages=messages,
            config=config, api_endpoint=api_endpoint,
            should_cancel=should_cancel, agent_kind=AGENT_KIND_PRIMARY,
            task=None, parent_run_id=None)
