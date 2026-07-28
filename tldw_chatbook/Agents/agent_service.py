# tldw_chatbook/Agents/agent_service.py
"""Wires the pure agent loop to the real provider, tools, and run store.

The ONLY impure Agents module: provider calls (chat_api_call), the
permission gate, sub-agent spawning, and AgentRunsDB persistence.
Runs synchronously — callers put it on a worker thread (Plan B).
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Callable, Protocol

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.Internal_Prompts.catalog import CATALOG
from tldw_chatbook.Utils.token_counter import count_tokens_messages, estimate_tokens

from .agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    RUN_DONE,
    RUN_ERROR,
    SPAWN_TOOL_NAME,
    STEP_ERROR,
    AgentConfig,
    AgentStep,
    ModelTurn,
    RunOutcome,
    SkillFileBindings,
    ToolCall,
    ToolResult,
    clamp_child_budget,
)
from .agent_runtime import LoopDeps, render_tool_protocol, run_agent_loop
from .native_tools import (
    ensure_tool_call_ids,
    parse_native_tool_calls,
    provider_supports_native_tools,
    schemas_to_openai_tools,
)
from .tool_catalog import (
    FIND_TOOLS_SCHEMA,
    INSTALL_SKILL_TOOL_SCHEMA,
    LOAD_TOOLS_SCHEMA,
    RUN_SKILL_SCRIPT_TOOL_SCHEMA,
    SEARCH_RUN_LOG_TOOL_SCHEMA,
    SKILL_FILE_TOOL_SCHEMA,
    SPAWN_TOOL_SCHEMA,
    ToolCatalogRegistry,
    initial_disclosure,
)

# Catalog-default re-export: keeps existing imports (console_agent_bridge,
# tests) valid and pins the "shipped default" used by the dual-prefix
# sub-agent check. Runtime call sites resolve live via get_internal_prompt.
SUBAGENT_SYSTEM_PROMPT = CATALOG["agents.subagent_system"].default

TRUNCATION_NOTICE = "\n[truncated]"

# Task 7: appended to config.system_prompt only when THIS run wired the
# search_run_log tool (see the `log_active` gate in _run_one, reused
# verbatim by _make_call_model) -- so the model is never told a log exists
# when it can't actually search it.
RUN_LOG_PROMPT_SECTION = (
    "Run log: every model turn, tool call, and tool result of this run is "
    "recorded in full to a log file. Your context holds a truncated view of "
    "it. When a result was truncated, or you need something from earlier in "
    "this run, call search_run_log to read the complete record instead of "
    "re-running the work or guessing. Prefer the 'contains' argument (a "
    "literal substring) over 'pattern' -- but note 'contains' and 'pattern' "
    "both match a record's CONTENT ONLY, never its metadata; use the "
    "'tool', 'type', 'status', and 'kind' arguments to filter by metadata. "
    "Search for specific content you know you need rather than browsing."
)


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

    def run(
        self, name: str, args: str, spawn: Callable[..., "ToolResult"]
    ) -> "ToolResult":
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


def _usage_total_tokens(resp) -> int | None:
    """Prompt+completion tokens from a provider's OpenAI-shaped usage block,
    or None when the provider didn't report usage.

    Args:
        resp: The provider response (dict when the provider reports usage).

    Returns:
        The total tokens for the call, or None to signal "estimate instead".
    """
    try:
        usage = resp["usage"]
    except (KeyError, TypeError):
        return None
    if not isinstance(usage, dict):
        return None
    # `type(x) is int` (not isinstance) rejects bool, which subclasses int;
    # require positive/non-negative real ints so a malformed usage block can't
    # corrupt or shrink the accumulated spend the runtime enforces on.
    total = usage.get("total_tokens")
    if type(total) is int and total > 0:
        return total
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        type(prompt) is int
        and type(completion) is int
        and prompt >= 0
        and completion >= 0
        and prompt + completion > 0
    ):
        return prompt + completion
    return None


_CANCEL_POLL_SECONDS = 0.5


def _call_with_timeout(
    fn: Callable[[], ToolResult],
    seconds: float,
    tool_name: str,
    should_cancel: Callable[[], bool] = lambda: False,
) -> ToolResult:
    """Run ``fn`` on a daemon thread, bounded by ``seconds`` wall-clock.

    Always returns a ToolResult: ``fn``'s value on success, ``ok=False`` with
    the message on a raised exception, or an ``ok=False`` timeout/cancelled
    result if ``fn`` does not finish in time or the run is cancelled first. A
    per-call daemon thread (NOT a ThreadPoolExecutor ``with`` block, whose
    __exit__ would join the hung worker and defeat the timeout; NOT a shared
    pool, which a single hung tool would saturate) is used; on timeout the
    worker is abandoned to die with the process — Python cannot forcibly kill
    a thread, but ``daemon`` means it never blocks interpreter shutdown, and
    for a side-effecting tool (notably an MCP call already past its own
    approval/execution bounds -- see ``RunBudget.max_tool_call_seconds``'s
    docstring) that abandoned worker may still complete and act for real
    after this function has already reported a timeout, so a caller retrying
    a "failed" call risks running it twice.

    The overall ``seconds`` wait is chunked into ``_CANCEL_POLL_SECONDS``
    slices so ``should_cancel`` is polled while a tool is hung, instead of a
    single blocking ``join(seconds)``: without this, a user pressing Stop
    during a slow/wedged tool call would see nothing until the FULL
    ``max_tool_call_seconds`` ceiling elapsed (300s at defaults) even though
    ``run_agent_loop`` itself is cooperative-cancel and re-checks
    ``should_cancel()`` at every call boundary -- the wait inside this one
    call was the one place that boundary couldn't be reached. A cancellation
    hit mid-wait is reported the same way a timeout is (``ok=False``,
    abandoned worker left to finish/die on its own), just with a "cancelled"
    message instead of "timed out" -- the abandon-on-timeout semantics and
    the overall ``seconds`` ceiling are both preserved unchanged.
    """
    box: dict = {}

    def _runner() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 — surfaced as a failed ToolResult, never propagated to the worker's exit
            box["error"] = str(exc)

    worker = threading.Thread(target=_runner, name=f"tool-{tool_name}", daemon=True)
    worker.start()
    deadline = time.monotonic() + seconds
    while worker.is_alive() and time.monotonic() < deadline:
        worker.join(min(_CANCEL_POLL_SECONDS, max(deadline - time.monotonic(), 0)))
        if worker.is_alive() and should_cancel():
            return ToolResult(ok=False, error=f"tool call cancelled: {tool_name}")
    if worker.is_alive():
        return ToolResult(
            ok=False, error=f"tool call timed out after {seconds:g}s: {tool_name}"
        )
    if "error" in box:
        return ToolResult(ok=False, error=box["error"])
    result = box.get("result")
    if result is None:
        return ToolResult(
            ok=False, error=f"tool call produced no result: {tool_name}"
        )
    return result


class AgentService:
    """Run one agent turn (primary + any sub-agents) and persist it."""

    def __init__(
        self,
        db: AgentRunsDB,
        registry: ToolCatalogRegistry,
        chat_call: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_step: Callable[[AgentStep, str], None] | None = None,
        skill_runner: SkillRunner | None = None,
        skill_file_bindings: SkillFileBindings | None = None,
        review_tool_calls: Callable[[list[ToolCall]], dict[str, str]] | None = None,
        review_state_scope: Callable[[], "contextlib.AbstractContextManager"]
        | None = None,
        install_skill_tool: Callable[[str], ToolResult] | None = None,
        run_skill_script_tool: Callable[[str, str, list[str]], ToolResult]
        | None = None,
        run_log_writer: "RunLogWriter | None" = None,
    ) -> None:
        self.db = db
        self.registry = registry
        self.chat_call = chat_call or _default_chat_call()
        self.clock = clock
        self._on_step = on_step
        self.skill_runner = skill_runner
        # task-3 (skills-foundation): per-run authorization + reader for the
        # skill_file runtime tool. `None` (the default, and every caller
        # before this task) means the run is never wired for skill_file at
        # all -- its schema is never pinned into runtime_schemas and a call
        # by that name falls through to normal unknown-tool handling (see
        # LoopDeps.read_skill_file's own docstring).
        self.skill_file_bindings = skill_file_bindings
        # P5 Task 4: generic pre-dispatch batch-review hook, threaded
        # straight into every LoopDeps this service builds (mirrors how
        # should_cancel flows through run_turn/_run_one). MCP-specific
        # wiring (Task 6) builds the callable passed here; this service
        # stays agnostic to what it does.
        self.review_tool_calls = review_tool_calls
        # C1 (probe-verified security regression, pre-merge review of the
        # Phase 5 chat bridge): an optional, generically-shaped seam a
        # caller can wire to snapshot/restore whatever per-turn state
        # `review_tool_calls` owns around a NESTED sub-agent run -- see
        # `spawn` below. `spawn_subagent` runs the child's entire loop
        # INLINE, synchronously, mid-parent-dispatch (before the parent's
        # own remaining same-batch tool calls are dispatched); if
        # `review_tool_calls` is backed by mutable shared state keyed only
        # by tool name (as `MCPToolProvider._stamped_decisions` is --
        # REPLACED, not merged, every turn including the child's own), the
        # child's turn(s) can silently clobber a verdict the PARENT's own
        # turn already decided, before the parent gets to consume it. `None`
        # (the default, and every caller before this task) preserves
        # byte-identical behavior: `spawn` falls back to a no-op
        # `contextlib.nullcontext()`. See `MCPToolProvider.stamp_scope` for
        # the concrete MCP-specific context manager wired here by
        # `console_agent_bridge.ConsoleAgentBridge.run_reply`.
        self.review_state_scope = review_state_scope
        # Agent-callable skill install (5th runtime tool). A ready-built
        # closure (enforce -> classify -> confirm -> install -> wrap) supplied
        # by the bridge. Pinned/wired ONLY for the top-level agent
        # (agent_kind == primary) in _run_one; a spawned subagent never gets
        # it. `None` (the default) means the run is not wired for install.
        self._install_skill_tool = install_skill_tool
        # Agent-callable skill script execution (6th runtime tool). All-agents
        # scope (spec §4.3): NO agent_kind gate, unlike install_skill above --
        # see the schema-pin comment in _run_one for the rationale. `None`
        # (the default) means the run is not wired for it.
        self._run_skill_script_tool = run_skill_script_tool
        # Round-1 review fix (spec §3.1): the writer is per RUN TREE, not
        # per service instance -- `bind()` latches permanently (see its own
        # docstring), so a writer built here in __init__ and reused across
        # two `run_turn` calls on the same `AgentService` would silently
        # append the second tree's records into the first tree's
        # already-bound directory and overwrite its manifest. `run_turn`
        # builds a fresh, UNBOUND writer per call from this attribute being
        # `None` -- see its own docstring/body. An explicitly injected
        # writer (tests, primarily) is the one exception: it is honored
        # as-is for the life of the service, on the assumption a caller
        # supplying its own writer also owns that writer's lifecycle.
        self._injected_run_log_writer = run_log_writer
        self.run_log_writer = run_log_writer

    # -- internals -------------------------------------------------------

    def _make_call_model(
        self,
        config: AgentConfig,
        api_endpoint: str,
        runtime_schemas: list,
        log_active: bool = False,
    ):
        native = config.native_tools and provider_supports_native_tools(api_endpoint)
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
            if log_active:
                system_content = f"{system_content}\n\n{RUN_LOG_PROMPT_SECTION}"
            payload = [{"role": "system", "content": system_content}]
            payload.extend(messages)
            resp = self.chat_call(
                api_endpoint=api_endpoint,
                messages_payload=payload,
                streaming=False,
                model=config.model,
                **call_kwargs,
            )
            text = _response_text(resp)
            tokens = _usage_total_tokens(resp)
            if tokens is None:
                # Provider reported no usage -> estimate from sent payload +
                # response text (native tool_calls JSON is not separately
                # counted here; the prompt term dominates the per-turn total).
                # Strip a provider prefix ("openai/gpt-4o-mini" -> "gpt-4o-mini")
                # so the tokenizer's model-family framing detection matches, and
                # pass the endpoint as the provider hint for the chars ratio.
                est_model = (
                    config.model.split("/", 1)[-1]
                    if "/" in config.model
                    else config.model
                )
                tokens = count_tokens_messages(
                    payload, est_model, provider=api_endpoint
                ) + estimate_tokens(text, est_model, provider=api_endpoint)
            if not native:
                return ModelTurn(text=text, tokens=tokens)
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
                    "role": "assistant",
                    "content": text,
                    "tool_calls": raw_calls,
                }
            return ModelTurn(
                text=text,
                tool_calls=tool_calls,
                assistant_message=assistant_message,
                tokens=tokens,
            )

        return call_model

    def _make_invoke_tool(
        self,
        config: AgentConfig,
        disclosed_names: set,
        should_cancel: Callable[[], bool] = lambda: False,
    ):
        def invoke_tool(call: ToolCall) -> ToolResult:
            if (
                call.name not in config.allowed_tools
                or call.name not in disclosed_names
            ):
                return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")
            timeout = self.registry.timeout_for(call.name) or (
                config.budget.max_tool_call_seconds
            )
            if timeout and timeout > 0:
                return _call_with_timeout(
                    lambda: self.registry.invoke_by_name(call.name, call.args),
                    timeout,
                    call.name,
                    should_cancel,
                )
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
        self.db.set_status(run_id, outcome.status, result=outcome.final_text or None)

    def _run_one(
        self,
        *,
        conversation_id: str,
        messages: list[dict],
        config: AgentConfig,
        api_endpoint: str,
        should_cancel: Callable[[], bool],
        agent_kind: str,
        task: str | None,
        parent_run_id: str | None,
        assistant_message_id: str | None = None,
    ) -> tuple[str, RunOutcome]:
        run_id = self.db.create_run(
            conversation_id=conversation_id,
            agent_kind=agent_kind,
            task=task,
            parent_run_id=parent_run_id,
            budget=dataclasses.asdict(config.budget),
            assistant_message_id=assistant_message_id,
        )
        # Two-phase: the writer was constructed before any run id existed.
        # Only the PRIMARY run binds; a child finds it already bound.
        if agent_kind == AGENT_KIND_PRIMARY:
            self.run_log_writer.bind(run_id)
        started = self.clock()

        active, offer_find_load = initial_disclosure(self.registry, config.budget)
        # Q7(a): the initial active set must respect the allow-list too —
        # the permission gate is a backstop, not the only checkpoint. A
        # disallowed tool must never even be disclosed to the model.
        active = [schema for schema in active if schema.name in config.allowed_tools]
        disclosed_names = {schema.name for schema in active}
        runtime_schemas = []
        if config.budget.max_subagents > 0:
            runtime_schemas.append(SPAWN_TOOL_SCHEMA)
        if offer_find_load:
            runtime_schemas.extend([FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA])
        if (
            self.skill_file_bindings is not None
            and self.skill_file_bindings.authorized
        ):
            runtime_schemas.append(SKILL_FILE_TOOL_SCHEMA)
        if agent_kind == AGENT_KIND_PRIMARY and self._install_skill_tool is not None:
            runtime_schemas.append(INSTALL_SKILL_TOOL_SCHEMA)
        # All-agents scope (spec §4.3): NO agent_kind gate. _run_one recurses
        # on this same service instance, so this intentionally reaches every
        # depth -- primary, skill forks, and spawned subagents alike. The gate
        # for each run is policy + trust + the confirm card / per-skill grant,
        # applied in the bridge closure and the service, not here.
        if self._run_skill_script_tool is not None:
            runtime_schemas.append(RUN_SKILL_SCRIPT_TOOL_SCHEMA)
        # search_run_log (7th runtime tool): primary agent only, like
        # install_skill above -- a depth-1 child's max_subagents is always
        # clamped to 0, so its "subtree" is only itself and its short
        # history is already in its context; letting it search would only
        # widen what it can see, into its parent's whole run tree, breaking
        # the isolation spawn_subagent promises. Also gated on the writer
        # actually being active: an inactive writer means no log directory
        # was ever created, so there is nothing to search.
        #
        # Placed LAST, after every other runtime_schemas append above, and
        # additionally requires `runtime_schemas or active` to be non-empty
        # (controller ruling, post-review of the original spec): unlike
        # every OTHER runtime tool -- spawn_subagent gated on
        # `max_subagents > 0`, find_tools/load_tools on `offer_find_load`,
        # skill_file on a non-empty authorized set -- an unconditional
        # `is_active` gate would offer this tool even to a run with
        # nothing else disclosed at all (empty allow-list, no sub-agents,
        # no skills). Such a run can only ever produce model-turn log
        # records -- it has no tool results, so nothing was ever
        # truncated and there is nothing to recover -- so the tool would
        # buy it nothing while changing the provider payload (adding a
        # `tools=` kwarg) of a deliberately tool-less run. See task-243
        # minor m3: a native-capable endpoint with no disclosable schemas
        # must send no `tools=` kwarg at all.
        # Task 7: reused verbatim (not re-derived) as the gate on whether the
        # system prompt's RUN_LOG_PROMPT_SECTION gets appended below, so the
        # prompt can never mention a tool this run didn't actually disclose.
        log_active = (
            agent_kind == AGENT_KIND_PRIMARY
            and self.run_log_writer.is_active
            and (runtime_schemas or active)
        )
        if log_active:
            runtime_schemas.append(SEARCH_RUN_LOG_TOOL_SCHEMA)

        def find_tools(query: str):
            # Q7(b): never surface a disallowed tool through find_tools,
            # even though it exists in the catalog.
            return [
                entry
                for entry in self.registry.find(query)
                if entry.name in config.allowed_tools
            ]

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
            accepted = schemas[: max(room, 0)]
            for schema in accepted:
                disclosed_names.add(schema.name)
            return accepted

        sub_agent_spawns = 0

        def spawn(
            spawn_task: str, *, allowed_tools: tuple[str, ...] | None = None
        ) -> ToolResult:
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
                return ToolResult(ok=False, error="sub-agent budget exhausted")
            sub_agent_spawns += 1
            remaining = config.budget.max_wall_seconds - (self.clock() - started)
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
                allowed_tools
                if allowed_tools is not None
                else tuple(
                    n
                    for n in config.allowed_tools
                    if n != SPAWN_TOOL_NAME
                    and not (
                        self.skill_runner is not None
                        and self.skill_runner.is_skill_tool(n)
                    )
                )
            )
            child_config = AgentConfig(
                model=config.model,
                system_prompt=get_internal_prompt("agents.subagent_system"),
                allowed_tools=child_allowed_tools,
                budget=clamp_child_budget(config.budget, remaining),
                native_tools=config.native_tools,
            )
            # C1: snapshot/restore whatever review_state_scope owns (see
            # __init__'s own comment) around the ENTIRE nested run -- the
            # child's own turns must never be able to leave the parent's
            # per-turn review state (e.g. MCPToolProvider._stamped_
            # decisions) mutated once control returns here. A no-op
            # contextlib.nullcontext() when no scope was wired (every
            # non-MCP run, and every caller before this task).
            scope = (
                self.review_state_scope()
                if self.review_state_scope
                else contextlib.nullcontext()
            )
            with scope:
                _child_id, child_outcome = self._run_one(
                    conversation_id=conversation_id,
                    messages=[{"role": "user", "content": spawn_task}],
                    config=child_config,
                    api_endpoint=api_endpoint,
                    should_cancel=should_cancel,
                    agent_kind=AGENT_KIND_SUBAGENT,
                    task=spawn_task,
                    parent_run_id=run_id,
                )
            text = child_outcome.final_text
            cap = config.budget.max_subagent_result_chars
            if len(text) > cap:
                text = text[:cap] + TRUNCATION_NOTICE
            if child_outcome.status != RUN_DONE:
                return ToolResult(
                    ok=False, error=f"sub-agent {child_outcome.status}: {text}"
                )
            return ToolResult(ok=True, content=text)

        # Skill-aware invoke_tool, built AFTER spawn (it closes over it): a
        # skill-tool call never reaches the registry/ToolProvider.invoke
        # path (SkillToolProvider.invoke raises by design -- Task 11 traced
        # that pre-wiring path as a loud full-run abort). Instead it routes
        # through skill_runner.run, which renders the skill and calls THIS
        # run's spawn -- so it is budget-counted (via spawn's own shared
        # sub_agent_spawns counter -- see Finding 2 above), cancellable, and
        # DB-lineage-tracked exactly like a spawn_subagent call.
        builtin_invoke_tool = self._make_invoke_tool(
            config, disclosed_names, should_cancel
        )

        def invoke_tool(call: ToolCall) -> ToolResult:
            if self.skill_runner is not None and self.skill_runner.is_skill_tool(
                call.name
            ):
                # Task-12 review Finding 1: a skill tool must pass the SAME
                # two-part gate as an ordinary catalog tool (mirrors
                # _make_invoke_tool above) -- allowed_tools is the
                # permission boundary, but disclosed_names (seeded by
                # initial disclosure and grown only via load_tools, exactly
                # like a builtin) is the other half. Checking allowed_tools
                # alone let an undisclosed skill name execute the instant
                # the model guessed it, even behind a >8-tool catalog where
                # progressive disclosure is supposed to gate exactly this.
                if (
                    call.name not in config.allowed_tools
                    or call.name not in disclosed_names
                ):
                    return ToolResult(
                        ok=False, error=f"Tool not permitted: {call.name}"
                    )
                # Cheap early exit before rendering the skill: the
                # authoritative check-and-increment lives in `spawn` itself
                # (shared with the native spawn_subagent path), so the
                # combined ceiling holds regardless of call order even
                # without this line -- it only saves an unnecessary
                # render/trust round-trip once the shared budget is spent.
                if sub_agent_spawns >= config.budget.max_subagents:
                    return ToolResult(ok=False, error="sub-agent budget exhausted")
                return self.skill_runner.run(
                    call.name, str(call.args.get("args", "")), spawn
                )
            return builtin_invoke_tool(call)

        # task-3 (skills-foundation): reader closure for the skill_file
        # runtime tool, built beside invoke_tool. Authorization is enforced
        # HERE (against self.skill_file_bindings.authorized), never in the
        # loop and never via config.allowed_tools -- see SkillFileBindings'
        # own docstring. The bindings-None guard below is defensive (the
        # LoopDeps wiring already gates this closure out entirely when no
        # bindings were passed to this service at all).
        def read_skill_file_tool(skill_name: str, path: str) -> ToolResult:
            bindings = self.skill_file_bindings
            if bindings is None or skill_name not in bindings.authorized:
                return ToolResult(
                    ok=False,
                    error=f"skill_file: '{skill_name}' is not active in this run",
                )
            if bindings.reader is None:
                return ToolResult(ok=False, error="skill_file: no reader configured")
            try:
                out = bindings.reader(skill_name, path)
                # task-4 (skills-fork-reachability) hardening: a reader is
                # caller-supplied (the bridge's asyncio.run adapter over
                # SkillsScopeService.read_skill_file) -- a misbehaving one
                # returning a non-mapping must fail only THIS call, not
                # crash the whole run via an uncaught AttributeError from
                # `.get` on something that isn't dict-like.
                if not isinstance(out, Mapping):
                    return ToolResult(
                        ok=False,
                        error="skill_file: reader returned invalid result",
                    )
                content = out.get("content", "")
            except Exception as exc:  # SkillTrustBlockedError, ValueError, OSError
                return ToolResult(ok=False, error=f"skill_file: {exc}")
            return ToolResult(ok=True, content=str(content))

        def search_run_log(args: dict) -> ToolResult:
            """Query THIS run's log. Reads only what this agent produced.

            F2 (Qodo #2, PR #1066 review -- DECLINED): Qodo's finding wanted
            these raw ``dict`` args routed through a Pydantic model before
            use. Declined: every OTHER runtime tool this service wires
            (``install_skill``, ``run_skill_script``, ``skill_file``) takes
            the exact same raw-dict-plus-defensive-cast shape, and every
            argument here is ALREADY coerced defensively (``str(...)`` for
            the metadata filters below; ``int(... or 0)`` inside a single
            ``try/except (TypeError, ValueError)`` for the numeric ones) --
            a bad value already returns a clean ``ToolResult`` error rather
            than raising, which is the property Pydantic would add. Giving
            this one tool a model would make it the odd one out in this
            module without changing behavior. See
            ``Tests/Agents/test_search_run_log_runtime_tool.py`` for the
            coverage confirming every argument is safely coerced (string
            where an int is expected, null, a nested object, a list).

            Args:
                args: The model-supplied call arguments, straight off
                    ``ToolCall.args`` (always a ``dict`` -- both parsing
                    paths in ``native_tools.py``/``agent_runtime.py``
                    guarantee that, never validated by a schema here).
                    Recognised keys mirror ``search_records``'/
                    ``format_results``' own parameters: ``contains``,
                    ``pattern``, ``tool``, ``type``, ``status``, ``kind``,
                    ``from_record``, ``to_record``, ``context``, ``offset``.

            Returns:
                ``ToolResult(ok=True, content=...)`` with the rendered hits
                (or "No matching records."), or ``ok=False`` with a
                human-readable error -- for a missing log, malformed
                numeric arguments, a rejected catastrophic-looking
                ``pattern``, or a search that exceeded its wall-clock
                budget (F6). Never raises.
            """
            from .run_log_search import (
                RunLogSearchPatternRejected,
                RunLogSearchTimeout,
                format_results,
                load_records,
                search_records,
            )

            log_dir = self.run_log_writer.log_dir
            if log_dir is None:
                return ToolResult(ok=False, error="No run log is available.")
            contains = str(args.get("contains", ""))
            pattern = str(args.get("pattern", ""))
            try:
                records = load_records(log_dir)
                hits = search_records(
                    records,
                    contains=contains,
                    pattern=pattern,
                    tool=str(args.get("tool", "")),
                    type=str(args.get("type", "")),
                    status=str(args.get("status", "")),
                    kind=str(args.get("kind", "")),
                    from_record=int(args.get("from_record") or 0),
                    to_record=int(args.get("to_record") or 0),
                    context=int(args.get("context") or 0),
                )
                # TASK-1250: offset, coerced the same defensively-numeric way
                # as from_record/to_record/context above -- a model sending
                # junk (a non-numeric string) is caught below like any other
                # bad numeric arg, never raised into the run. Negative and
                # past-the-end clamping happens in format_results itself
                # (single point of truth, mirroring `context`'s own clamp).
                offset = int(args.get("offset") or 0)
            except (TypeError, ValueError) as exc:
                return ToolResult(ok=False, error=f"Invalid search arguments: {exc}")
            except (RunLogSearchPatternRejected, RunLogSearchTimeout) as exc:
                # F6 (Qodo #6): a model-supplied `pattern=` that looks
                # catastrophic, or a search that ran past its wall-clock
                # budget -- both must degrade to a normal tool error, never
                # raise into (and abort) this run. See run_log_search.py's
                # module docstring for why neither defense is complete
                # alone.
                return ToolResult(ok=False, error=str(exc))
            # Final-review CRITICAL 1: render recovered records at THIS run's
            # own tool-result ceiling, not format_results' 400-char rendering
            # default. §6.1's whole point is that a truncation trailer points
            # at a lossless copy -- if the copy renders at 400 chars while
            # the truncation it repairs cut at 16,000, following the trailer
            # returns LESS than the thing it was supposed to fix. Keep
            # format_results' own default untouched (its existing tests pin
            # 400) and instead pass the run's actual ceiling from here.
            # 0 (or negative) is the documented "unlimited" value for
            # RunBudget.max_tool_result_chars / _truncate_tool_result;
            # format_results has no such sentinel (max_chars=0 would render
            # nothing), so translate it into a ceiling that never trips
            # format_results' own truncation branch. The rendered search
            # RESULT still passes back through the loop's ordinary
            # _truncate_tool_result at the history-append seam, so this
            # cannot blow the run's context budget -- it only stops the
            # recovery path from being strictly worse than what it repairs.
            #
            # TASK-1250: that alone was not enough. format_results always
            # rendered from character 0 of each record, which is the SAME
            # ceiling that truncated the result in the first place -- so a
            # record larger than render_max_chars still rendered byte-
            # identical to what history already showed, and a `contains=`
            # match past that ceiling could render a body that did not
            # contain it. Passing `contains`/`pattern` lets format_results
            # centre the window on the actual match; passing `offset` lets
            # the model page past render_max_chars deterministically once a
            # render tells it the next offset to use.
            ceiling = config.budget.max_tool_result_chars
            render_max_chars = ceiling if ceiling > 0 else sys.maxsize
            return ToolResult(
                ok=True,
                content=format_results(
                    hits,
                    max_chars=render_max_chars,
                    contains=contains,
                    pattern=pattern,
                    offset=offset,
                ),
            )

        def on_record(record_type: str, payload: dict) -> int | None:
            """Append one full-fidelity record to THIS run tree's log.

            The ``LoopDeps.on_record`` callable: called by
            ``agent_runtime.run_agent_loop`` (via its ``_emit_record``
            helper) at the two points the COMPLETE value exists, before any
            truncation. Wraps ``self.run_log_writer.append`` with this
            run's identity (``run_id``, ``agent_kind``) and defensively
            stringifies every payload field, so a malformed payload can
            never raise here either.

            Args:
                record_type: ``"model"``, ``"tool_call"``, or
                    ``"tool_result"`` (``_emit_record``'s own vocabulary;
                    ``"spawn"`` is not currently emitted -- a spawn's
                    dispatch is captured as an ordinary ``tool_call``/
                    ``tool_result`` pair like any other tool).
                payload: ``content``/``tool``/``status``/``call_id``, as
                    built by ``_emit_record``'s ``**payload`` kwargs.

            Returns:
                The record number MUST be returned here, not swallowed:
                Task 7 threads it into the truncation trailer so a cut
                result points at its own full copy in the log (see
                ``_truncate_tool_result``). ``None`` when the writer is
                inactive or the underlying write failed -- never raises.
            """
            return self.run_log_writer.append(
                run_id=run_id,
                kind=agent_kind,
                type=record_type,
                content=str(payload.get("content", "")),
                tool=str(payload.get("tool", "")),
                status=str(payload.get("status", "")),
                call_id=str(payload.get("call_id", "")),
            )

        deps = LoopDeps(
            call_model=self._make_call_model(
                config, api_endpoint, runtime_schemas, log_active
            ),
            invoke_tool=invoke_tool,
            spawn=spawn,
            find_tools=find_tools,
            load_schemas=load_schemas,
            should_cancel=should_cancel,
            clock=self.clock,
            on_step=(
                (lambda s: self._on_step(s, agent_kind))
                if self._on_step is not None
                else (lambda s: None)
            ),
            review_tool_calls=self.review_tool_calls,
            # Qodo/PR#814: wired under the SAME predicate as the schema pin
            # above (~:356-360) -- bindings with an EMPTY authorized set
            # must never reach the named-refusal dispatch either; a
            # hallucinated call for an unpinned tool falls through to the
            # generic "Tool not permitted" path like any other undisclosed
            # tool name.
            read_skill_file=(
                read_skill_file_tool
                if self.skill_file_bindings is not None
                and self.skill_file_bindings.authorized
                else None
            ),
            install_skill=(
                self._install_skill_tool
                if agent_kind == AGENT_KIND_PRIMARY
                and self._install_skill_tool is not None
                else None
            ),
            run_skill_script=self._run_skill_script_tool,
            search_run_log=(
                search_run_log if agent_kind == AGENT_KIND_PRIMARY else None
            ),
            on_record=on_record,
        )
        try:
            outcome = run_agent_loop(config, messages, active, deps)
        except Exception as exc:  # noqa: BLE001 — a run never raises out
            from tldw_chatbook.Chat.provider_failures import describe_stream_failure

            # TASK-335: raw str(exc) is httpx's status line + MDN boilerplate;
            # the classified copy carries the provider's response-body message
            # instead — this summary becomes user-facing failure copy.
            outcome = RunOutcome(
                status=RUN_ERROR,
                steps=[
                    AgentStep(
                        index=0,
                        kind=STEP_ERROR,
                        summary=describe_stream_failure(exc)[:500],
                    )
                ],
            )
        self._persist(run_id, outcome)
        return run_id, outcome

    # -- public ----------------------------------------------------------

    def run_turn(
        self,
        *,
        conversation_id: str,
        messages: list[dict],
        config: AgentConfig,
        api_endpoint: str,
        should_cancel: Callable[[], bool] = lambda: False,
        supersede_run_id: str | None = None,
        assistant_message_id: str | None = None,
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
            assistant_message_id: Recorded on the primary run at creation
                time (only the primary run — never a spawned sub-agent,
                which produces no transcript reply). At create time this is
                the reply's NATIVE in-memory id; the assistant node is not
                persisted yet, so the caller overwrites it with the durable
                persisted id via ``AgentRunsDB.set_run_assistant_message_id``
                once the reply completes (that later write is what resume's
                marker anchoring reads).

        Returns:
            A ``(run_id, outcome)`` tuple: the new primary run's id and its
            terminal ``RunOutcome``. The run record (and any sub-agent run
            records) are persisted before this returns.

        Run-log contract:
            The run-log writer is scoped to ONE run tree, not to this
            service instance. Unless a writer was explicitly injected via
            the constructor (tests, primarily — that one is reused as-is
            for the life of the service), each call to ``run_turn`` builds
            a fresh, unbound ``RunLogWriter``. ``bind()`` latches
            permanently for that writer's whole life (see its own
            docstring), so reusing one writer across two ``run_turn`` calls
            would append the second tree's records into the first tree's
            already-bound directory and overwrite its manifest.
        """
        if supersede_run_id:
            self.db.supersede_run_tree(supersede_run_id)
        # Per run tree, not per service instance -- see "Run-log contract"
        # above. `_injected_run_log_writer` is `None` for every caller that
        # didn't pass one to the constructor (i.e. every production caller
        # today), so this builds a new, unbound writer each call; an
        # injected writer is honored unchanged.
        if self._injected_run_log_writer is not None:
            self.run_log_writer = self._injected_run_log_writer
        else:
            from .run_log import RunLogWriter as _RunLogWriter

            self.run_log_writer = _RunLogWriter()
        # Per-run scope for the registry's owner-map cache (tool_catalog's
        # _owner_and_id): reset here, once, at the top of the run tree —
        # covers the primary turn AND any sub-agents it spawns via
        # _run_one, since they never call run_turn themselves. The catalog
        # is listed fresh at this point, so skill CRUD since the last run
        # is always picked up with no separate invalidation signal needed.
        self.registry.reset_catalog_cache()
        run_id, outcome = self._run_one(
            conversation_id=conversation_id,
            messages=messages,
            config=config,
            api_endpoint=api_endpoint,
            should_cancel=should_cancel,
            agent_kind=AGENT_KIND_PRIMARY,
            task=None,
            parent_run_id=None,
            assistant_message_id=assistant_message_id,
        )
        # Manifest needs run-level metadata the writer itself does not have
        # (including supersession), so it is written once the whole run
        # tree finishes, here rather than inside _run_one.
        self.run_log_writer.write_manifest(
            {
                "run_id": run_id,
                "model": config.model,
                "api_endpoint": api_endpoint,
                "allowed_tools": list(config.allowed_tools),
                "budget": dataclasses.asdict(config.budget),
                "status": outcome.status,
                "superseded_run_id": supersede_run_id or "",
                "total_tokens": outcome.total_tokens,
            }
        )
        self.run_log_writer.close()
        return run_id, outcome
