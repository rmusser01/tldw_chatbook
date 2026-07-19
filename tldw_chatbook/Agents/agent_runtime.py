"""Pure agent control loop + fence-first text tool protocol.

No Textual, app, DB, or I/O imports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from loguru import logger

from .agent_models import (
    FIND_TOOLS_NAME, LOAD_TOOLS_NAME, LOOP_DETECTION_N, RUN_CANCELLED,
    RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME, STEP_ERROR, STEP_MODEL,
    STEP_SPAWN, STEP_TOOL_CALL, STEP_TOOL_RESULT, AgentConfig, AgentStep,
    ModelTurn, RunOutcome, ToolCall, ToolResult, ToolSchema,
)

FENCE_OPEN = "```tool_call"
_FENCE_CLOSE = "```"

STREAM_TOOL_CALL = "tool_call"
STREAM_TEXT = "text"
STREAM_UNDECIDED = "undecided"


def parse_fenced_tool_call(text: str) -> ToolCall | None:
    """Parse a response whose FIRST non-whitespace content is a tool fence.

    Args:
        text: The full model response text to inspect.

    Returns:
        The parsed ``ToolCall`` if the leading fence is a well-formed
        ```tool_call`` block containing JSON with a string ``name`` and an
        ``arguments`` object; ``None`` for anything malformed (never
        raises).
    """
    stripped = text.lstrip()
    if not stripped.startswith(FENCE_OPEN):
        return None
    after = stripped[len(FENCE_OPEN):]
    newline = after.find("\n")
    if newline == -1:
        return None
    # The character right after FENCE_OPEN must end the tag line (only
    # whitespace may follow before the newline). Otherwise this is a
    # look-alike tag such as ```tool_calls or ```tool_call_schema, not a
    # real fence, and must not be parsed as a tool call.
    tag_line_rest = after[:newline]
    if tag_line_rest.strip():
        return None
    body_and_rest = after[newline + 1:]
    close = body_and_rest.find(_FENCE_CLOSE)
    if close == -1:
        return None
    raw = body_and_rest[:close].strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    args = payload.get("arguments", {})
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        return None
    return ToolCall(name=name, args=args)


def split_visible_text_and_tool_call(text: str) -> tuple[str, ToolCall | None]:
    """Handle a disobedient mid-stream fence: visible prefix + parsed call.

    No fence, or a fence that does not parse → the full text stays visible
    and the call is None.

    A look-alike tag (```tool_calls, ```tool_call_schema, ...) that merely
    starts with FENCE_OPEN does not parse, so scan forward past it to see
    if a real fence follows later in the text.

    Args:
        text: The full model response text to inspect.

    Returns:
        A ``(visible_text, tool_call)`` tuple. If no fence is found, or a
        fence is found but does not parse, ``visible_text`` is the full
        input and ``tool_call`` is ``None``. Otherwise ``visible_text`` is
        the text preceding the fence (right-stripped) and ``tool_call`` is
        the parsed call.
    """
    start = 0
    while True:
        idx = text.find(FENCE_OPEN, start)
        if idx == -1:
            return text, None
        call = parse_fenced_tool_call(text[idx:])
        if call is not None:
            return text[:idx].rstrip(), call
        start = idx + len(FENCE_OPEN)


def stream_prefix_verdict(prefix: str) -> str:
    """Sniff a stream's first tokens: tool_call, text, or undecided.

    Args:
        prefix: The tokens received so far from a streaming response.

    Returns:
        One of ``STREAM_TOOL_CALL``, ``STREAM_TEXT``, or
        ``STREAM_UNDECIDED`` — the latter meaning more tokens are needed
        before a verdict can be reached (e.g. the prefix could still grow
        into a look-alike fence tag).
    """
    stripped = prefix.lstrip()
    if not stripped:
        return STREAM_UNDECIDED
    if stripped.startswith(FENCE_OPEN):
        # Matching FENCE_OPEN alone is not decisive: the stream could still
        # grow into a look-alike tag like ```tool_calls or
        # ```tool_call_schema. Only a clean line boundary (whitespace then
        # newline) after FENCE_OPEN confirms a real tool-call fence.
        after = stripped[len(FENCE_OPEN):]
        if not after:
            return STREAM_UNDECIDED
        i = 0
        while i < len(after) and after[i] in (" ", "\t"):
            i += 1
        if i == len(after):
            # Nothing but trailing spaces so far — could still become a
            # newline (tool_call) or more characters (look-alike tag).
            return STREAM_UNDECIDED
        if after[i] in ("\n", "\r"):
            return STREAM_TOOL_CALL
        return STREAM_TEXT
    if FENCE_OPEN.startswith(stripped):
        return STREAM_UNDECIDED
    return STREAM_TEXT


def render_tool_protocol(schemas: list[ToolSchema]) -> str:
    """Render the tool-protocol system-prompt section.

    Args:
        schemas: The tool schemas currently disclosed to the model.

    Returns:
        The protocol instructions plus a JSON rendering of each schema, or
        an empty string when ``schemas`` is empty (no protocol section:
        the model should answer directly).
    """
    if not schemas:
        return ""
    blocks = []
    for schema in schemas:
        blocks.append(json.dumps(
            {"name": schema.name, "description": schema.description,
             "parameters": schema.parameters}, indent=2))
    tool_list = "\n".join(blocks)
    return (
        "You can call tools. Available tools:\n"
        f"{tool_list}\n\n"
        "To call a tool, your reply MUST START with the fence as its first "
        "content — no prose before it:\n"
        f'{FENCE_OPEN}\n{{"name": "<tool name>", "arguments": {{...}}}}\n'
        f"{_FENCE_CLOSE}\n"
        "One tool call per reply. After you receive the tool result, either "
        "call another tool the same way or answer the user directly. If no "
        "tool is needed, just answer directly."
    )


@dataclass
class LoopDeps:
    """Everything impure, injected. The loop itself stays pure."""

    call_model: Callable[[list, tuple], ModelTurn]
    invoke_tool: Callable[..., ToolResult]
    # Callable[..., ...] (not Callable[[str], ...]): the loop itself only
    # ever calls spawn(task) positionally, but the real implementation
    # (agent_service._run_one's spawn closure) also accepts a keyword-only
    # `allowed_tools` override, used by the skill-tool dispatch path
    # (SkillRunner.run) to narrow a spawned child's allow-list. The loop
    # never passes it and never needs to.
    spawn: Callable[..., ToolResult]
    find_tools: Callable[[str], list]
    load_schemas: Callable[[list], list]
    should_cancel: Callable[[], bool]
    clock: Callable[[], float]
    on_step: Callable[[AgentStep], None] = lambda step: None
    # Optional pre-dispatch batch-review hook (P5 Task 4): the generic seam
    # the MCP approval flow (Task 6) rides on. When set, called ONCE per
    # turn with the full batch of tool calls about to be dispatched
    # (native multi-call batch, or the single fence-parsed call), BEFORE
    # any of them is invoked. Returns a name -> verdict map; "proceed"
    # (or an absent name -- a call the hook doesn't mention is presumed
    # fine) dispatches normally, anything else is treated as a refusal
    # string that is fed back to the model as that call's tool result
    # instead of invoking it. Exceptions are caught, logged, and treated
    # as "proceed" for every call in the batch -- the hook fails OPEN
    # here; MCP-specific fail-closed behavior lives inside the Task 6
    # closure, not in this generic runtime. ``None`` (the default) is a
    # no-op: every call proceeds, byte-identical to pre-Task-4 behavior.
    review_tool_calls: Callable[[list[ToolCall]], dict[str, str]] | None = None


def _catalog_lines(entries: list) -> str:
    if not entries:
        return "No matching tools."
    return "\n".join(
        f"{e.id} — {e.name}: {e.one_line_description}" for e in entries)


def _append_tool_result(messages: list[dict], call: ToolCall,
                        content: str) -> None:
    """Append one tool result to history using the call's role/id shaping.

    Single source of truth for both the normal post-invoke path and the
    review-hook refusal path (P5 Task 4) so the two can never drift.

    Native protocol (``call.call_id`` set): a ``role="tool"`` message
    paired to the assistant turn's ``tool_calls`` entry by
    ``tool_call_id``. Fence protocol (``call.call_id`` unset): the
    plain-text ``"Tool result for {name}: {content}"`` convention,
    appended as a user-role message.
    """
    if call.call_id:
        messages.append({
            "role": "tool", "tool_call_id": call.call_id,
            "content": content})
    else:
        messages.append({
            "role": "user",
            "content": f"Tool result for {call.name}: {content}"})


def run_agent_loop(config: AgentConfig, initial_messages: list[dict],
                   active_schemas: list, deps: LoopDeps) -> RunOutcome:
    """Drive think → (tool) → observe until done / stuck / cancelled.

    Message convention (transport-independent): fence-protocol turns append
    the assistant text verbatim and tool results append as user-role
    ``Tool result for {name}: {content}`` lines; native tool-call turns
    (``call.call_id`` set) instead append the provider-shaped
    ``turn.assistant_message`` echo and pair each tool result to its call
    as a ``role="tool"`` message keyed on ``tool_call_id``.

    Args:
        config: The agent's model, system prompt, allow-list, and budget
            (step count, wall-clock seconds, and — task-244 —
            provider-call/model-turn count all independently cap the run).
        initial_messages: The starting conversation history (role/content
            dicts); not mutated in place — the loop works on a copy.
        active_schemas: Tool schemas already disclosed to the model at the
            start of the run (may be empty when disclosure is deferred to
            ``find_tools``/``load_tools``).
        deps: The injected impure callables (provider call, tool
            invocation, spawn, tool discovery/loading, cancellation
            check, clock).

    Returns:
        A ``RunOutcome`` capturing the terminal status
        (``done``/``stuck``/``cancelled``), the full step log, the final
        answer text (when done), and how many sub-agents were spawned.
    """
    budget = config.budget
    steps: list[AgentStep] = []
    messages = list(initial_messages)
    active = list(active_schemas)
    started = deps.clock()
    spawned = 0
    model_turns = 0
    last_key: tuple | None = None
    repeat_count = 0

    def add(kind: str, **kw) -> AgentStep:
        step = AgentStep(index=len(steps), kind=kind, **kw)
        steps.append(step)
        # The hook drives live UI only (see LoopDeps.on_step docstring);
        # durability comes from the service's end-of-run persist, so a
        # raising callback must never abort or corrupt the run itself.
        try:
            deps.on_step(step)
        except Exception:  # noqa: BLE001 — best-effort UI notification only
            pass
        return step

    while True:
        if deps.should_cancel():
            return RunOutcome(RUN_CANCELLED, steps,
                              subagents_spawned=spawned)
        if len(steps) >= budget.max_steps:
            add(STEP_ERROR, summary="step budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)
        if model_turns >= budget.max_model_turns:
            add(STEP_ERROR, summary="model-turn budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)
        if deps.clock() - started > budget.max_wall_seconds:
            add(STEP_ERROR, summary="wall-clock budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)

        turn = deps.call_model(messages, tuple(active))
        model_turns += 1
        add(STEP_MODEL, summary=turn.text[:200])

        calls = list(turn.tool_calls)
        if not calls:
            _visible, fenced = split_visible_text_and_tool_call(turn.text)
            if fenced is None:
                # A Stop can land while this (tool-call-free) turn was still
                # streaming. There is no further step/tool-call boundary
                # ahead for a plain final answer, so this is the last chance
                # to recheck should_cancel before reporting "done" — without
                # this, a cancellation that lands mid-final-answer would be
                # silently downgraded to a normal completed run.
                if deps.should_cancel():
                    return RunOutcome(RUN_CANCELLED, steps,
                                      final_text=turn.text,
                                      subagents_spawned=spawned)
                return RunOutcome(RUN_DONE, steps, final_text=turn.text,
                                  subagents_spawned=spawned)
            calls = [fenced]
        messages.append(turn.assistant_message
                        or {"role": "assistant", "content": turn.text})

        # P5 Task 4: optional pre-dispatch batch review, called ONCE with
        # the full batch about to be dispatched below (whichever produced
        # it -- native multi-call or the single fence-parsed call) and
        # BEFORE any of them is invoked. `deps.review_tool_calls is None`
        # (the default) short-circuits to an empty verdicts map, which
        # makes every `.get(name, "proceed")` lookup below resolve to
        # "proceed" -- the exact same dispatch path as before this hook
        # existed, so absent-hook behavior stays byte-identical.
        verdicts: dict[str, str] = {}
        if deps.review_tool_calls is not None and calls:
            try:
                verdicts = deps.review_tool_calls(list(calls)) or {}
            except Exception as exc:  # noqa: BLE001 — fail OPEN here; the
                # MCP-specific fail-closed policy lives in the Task 6
                # closure that builds this callable, not in this generic
                # runtime.
                logger.opt(exception=True).warning(
                    f"review_tool_calls hook raised for batch "
                    f"{[c.name for c in calls]}; treating all {len(calls)} "
                    f"calls as proceed")
                verdicts = {}

        for call in calls:
            if deps.should_cancel():
                return RunOutcome(RUN_CANCELLED, steps,
                                  subagents_spawned=spawned)
            key = (call.name, json.dumps(call.args, sort_keys=True))
            repeat_count = repeat_count + 1 if key == last_key else 1
            last_key = key
            if repeat_count >= LOOP_DETECTION_N:
                add(STEP_ERROR,
                    summary=f"loop detected: {call.name} repeated "
                            f"{repeat_count}x with identical args")
                return RunOutcome(RUN_STUCK, steps,
                                  subagents_spawned=spawned)

            # P5 Task 4: a non-"proceed" verdict (an absent name defaults to
            # "proceed" — the hook only reports what it wants to stop)
            # skips dispatch entirely: none of the SPAWN/find_tools/
            # load_tools/invoke_tool branches below run, and the verdict
            # string itself becomes the call's tool result, same as any
            # other result content from here down.
            # NOTE: verdict lookup by name only; same-name calls in one batch
            # share a verdict (T5/T6 closure authors: this is a known limitation).
            verdict = verdicts.get(call.name, "proceed")
            if verdict != "proceed":
                content = verdict
            else:
                if call.name == SPAWN_TOOL_NAME:
                    if SPAWN_TOOL_NAME not in config.allowed_tools:
                        # Q6: refuse before dispatch — no budget consumption,
                        # no STEP_SPAWN, deps.spawn never called.
                        result = ToolResult(
                            ok=False,
                            error=f"Tool not permitted: {SPAWN_TOOL_NAME}")
                    else:
                        task = str(call.args.get("task", "")).strip()
                        if not task:
                            # G4: an empty task is refused with no budget
                            # consumption and no STEP_SPAWN.
                            result = ToolResult(
                                ok=False,
                                error="Task description cannot be empty")
                        elif spawned >= budget.max_subagents:
                            result = ToolResult(
                                ok=False, error="sub-agent budget exhausted")
                        else:
                            add(STEP_SPAWN, summary=task[:200],
                                tool_name=SPAWN_TOOL_NAME, args=dict(call.args))
                            result = deps.spawn(task)
                            spawned += 1
                elif call.name == FIND_TOOLS_NAME:
                    add(STEP_TOOL_CALL, tool_name=call.name,
                        args=dict(call.args))
                    entries = deps.find_tools(str(call.args.get("query", "")))
                    result = ToolResult(ok=True, content=_catalog_lines(entries))
                elif call.name == LOAD_TOOLS_NAME:
                    add(STEP_TOOL_CALL, tool_name=call.name,
                        args=dict(call.args))
                    # G1/Q9: `ids` may legitimately arrive as a bare string
                    # (one id) or as None/other junk from an unreliable local
                    # model — never crash, and never char-split a string.
                    raw_ids = call.args.get("ids")
                    if isinstance(raw_ids, str):
                        ids = [raw_ids]
                    elif isinstance(raw_ids, list):
                        ids = [str(x) for x in raw_ids]
                    else:
                        ids = []
                    loaded = deps.load_schemas(ids)
                    if not loaded:
                        # G5: every id was invalid (or none were valid) — this
                        # is a different failure than "valid but no room".
                        result = ToolResult(
                            ok=False, error="No valid tools found to load")
                    else:
                        # F1-b (plan-a-final-review addendum): a provider may
                        # legitimately hand back a schema whose name is
                        # already in `active` (a re-load of an already-active
                        # tool). Drop those here, BEFORE the room slice below,
                        # so `active` can never gain a duplicate name even if
                        # a caller-side gate (e.g. agent_service's
                        # disclosed_names filtering) is bypassed or desyncs —
                        # this is the loop's own last line of defense for its
                        # list-vs-set cap-boundary integrity.
                        active_names = {a.name for a in active}
                        already_active = [s.name for s in loaded if s.name in active_names]
                        # PR #655 review: also dedupe by name WITHIN this batch
                        # (a caller may hand back the same schema twice — e.g.
                        # bare name + catalog id aliases) so `active` can never
                        # gain a duplicate from one load, mirroring the
                        # across-rounds guard above.
                        new_loaded = []
                        batch_names: set = set()
                        for s in loaded:
                            if s.name in active_names or s.name in batch_names:
                                continue
                            batch_names.add(s.name)
                            new_loaded.append(s)
                        if not new_loaded:
                            # Every requested id was already active — a no-op,
                            # not the "no valid ids at all" error case above,
                            # and (Gemini M, PR #636 bot review) not the same
                            # "no room" message a genuinely budget-exhausted
                            # request gets below: those two reasons a load
                            # accepts nothing are different for the model to
                            # act on (proceed to just call the tool it already
                            # has vs. it must free room first), so they must
                            # not read identically.
                            result = ToolResult(
                                ok=True,
                                content="already loaded: " + ", ".join(already_active))
                        else:
                            room = budget.max_active_tools - len(active)
                            accepted = new_loaded[:max(room, 0)]
                            active.extend(accepted)
                            if accepted:
                                result = ToolResult(
                                    ok=True,
                                    content="loaded: " + ", ".join(
                                        s.name for s in accepted))
                            else:
                                result = ToolResult(ok=True, content="no room")
                else:
                    add(STEP_TOOL_CALL, tool_name=call.name,
                        args=dict(call.args))
                    result = deps.invoke_tool(call)

                content = result.content if result.ok else f"ERROR: {result.error}"

            add(STEP_TOOL_RESULT, tool_name=call.name,
                result=content[:2000])
            _append_tool_result(messages, call, content)
