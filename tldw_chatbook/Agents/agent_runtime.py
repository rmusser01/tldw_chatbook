"""Pure agent control loop + fence-first text tool protocol.

No Textual, app, DB, or I/O imports.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from loguru import logger

from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRestoreTarget,
    ContinuationResult,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    transition_provider_call,
    validate_continuation_restore,
)
from tldw_chatbook.model_capabilities import (
    moonshot_model_returns_reasoning_content,
)

from .agent_models import (
    CHECK_AGENTS_TOOL_NAME,
    FENCE_TOOL_RESULT_PREFIX,
    FIND_TOOLS_NAME,
    INSTALL_SKILL_TOOL_NAME,
    LOAD_TOOLS_NAME,
    LOOP_DETECTION_N,
    MAX_LOOP_PERIOD,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    RUN_LOG_SLICE_TOOL_NAME,
    RUN_LOG_STATS_TOOL_NAME,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    RUN_STUCK,
    SEARCH_RUN_LOG_TOOL_NAME,
    SEND_TO_AGENT_TOOL_NAME,
    SKILL_FILE_TOOL_NAME,
    SPAWN_TOOL_NAME,
    STEP_ERROR,
    STEP_MODEL,
    STEP_MODEL_CANCELLED,
    STEP_MODEL_ERROR,
    STEP_MODEL_REQUEST_STARTED,
    STEP_MODEL_RESPONSE_COMPLETED,
    STEP_MODEL_RETRY,
    STEP_APPROVAL_APPROVED,
    STEP_APPROVAL_DENIED,
    STEP_APPROVAL_REQUESTED,
    STEP_APPROVAL_REVOKED,
    STEP_SPAWN,
    STEP_STEERING,
    STEP_TOOL_CALL,
    STEP_TOOL_CANCELLED,
    STEP_TOOL_EXECUTION_STARTED,
    STEP_TOOL_FAILED,
    STEP_TOOL_PROPOSED,
    STEP_TOOL_RESULT,
    STEP_TOOL_SUCCEEDED,
    STEP_TOOL_TIMED_OUT,
    TOOL_OUTCOME_BLOCKED,
    TOOL_OUTCOME_FAILED,
    TOOL_OUTCOME_SUCCESS,
    TOOL_OUTCOME_CANCELLED,
    TOOL_OUTCOME_TIMEOUT,
    WAIT_AGENTS_TOOL_NAME,
    AgentConfig,
    AgentStep,
    ContinuationEventContext,
    FinalContinuation,
    ModelTurn,
    ProviderContinuationEvent,
    RunOutcome,
    ToolBatchReady,
    ToolCall,
    ToolCallExecuting,
    ToolCallFinished,
    ToolResult,
    ToolSchema,
    format_steering_message,
)
from .project_instruction_runtime import (
    PROJECT_INSTRUCTION_ROW_KEY,
    InstructionChainPayloadState,
    InstructionDeliveryReceipt,
    build_project_instruction_deferral_rows,
)


def _utc_now() -> datetime:
    """Return the UTC wall clock used to stamp durable agent steps."""
    return datetime.now(timezone.utc)


def safe_utc_timestamp(wall_clock: Callable[[], datetime]) -> str:
    """Read an injected wall clock without making step capture load-bearing."""
    try:
        value = wall_clock()
        if not isinstance(value, datetime):
            raise TypeError("wall clock must return datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("wall clock must return an aware datetime")
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception:  # noqa: BLE001 — timestamp capture is best-effort
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


FENCE_OPEN = "```tool_call"
_FENCE_CLOSE = "```"

STREAM_TOOL_CALL = "tool_call"
STREAM_TEXT = "text"
STREAM_UNDECIDED = "undecided"


def _noop_provider_continuation(event: ProviderContinuationEvent) -> None:
    """Preserve legacy callers that never opt into provider continuation."""


@dataclass(frozen=True, slots=True)
class ToolBatchPreparation:
    """Typed result of preparing one complete tool-call batch."""

    status: Literal["proceed", "retry_with_context"]
    ephemeral_rows: tuple[Mapping[str, Any], ...] = ()
    delivery_receipt: InstructionDeliveryReceipt | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "ephemeral_rows", tuple(self.ephemeral_rows))
        except TypeError as error:
            raise ValueError("invalid tool batch preparation") from error
        if self.status == "proceed":
            if self.ephemeral_rows or self.delivery_receipt is not None:
                raise ValueError("proceed preparation cannot carry context")
            return
        if (
            self.status != "retry_with_context"
            or not self.ephemeral_rows
            or self.delivery_receipt is None
        ):
            raise ValueError("retry preparation requires rows and a receipt")
        row_keys = tuple(
            row.get(PROJECT_INSTRUCTION_ROW_KEY) for row in self.ephemeral_rows
        )
        if row_keys != self.delivery_receipt.row_keys or any(
            row.get(EPHEMERAL_ORIGIN_KEY) != "project_instructions"
            for row in self.ephemeral_rows
        ):
            raise ValueError("instruction receipt does not match context rows")


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
    after = stripped[len(FENCE_OPEN) :]
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
    body_and_rest = after[newline + 1 :]
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
    raw_arguments = ""
    decoder = json.JSONDecoder()
    try:
        cursor = 1
        while cursor < len(raw):
            while cursor < len(raw) and (raw[cursor].isspace() or raw[cursor] == ","):
                cursor += 1
            if cursor >= len(raw) or raw[cursor] == "}":
                break
            key, cursor = decoder.raw_decode(raw, cursor)
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
            if raw[cursor] != ":":
                return None
            cursor += 1
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
            value_start = cursor
            _value, cursor = decoder.raw_decode(raw, cursor)
            if key == "arguments":
                raw_arguments = raw[value_start:cursor]
    except (IndexError, TypeError, ValueError):
        return None
    call_id = payload.get("call_id", "")
    if not isinstance(call_id, str):
        return None
    return ToolCall(
        name=name,
        args=args,
        call_id=call_id,
        raw_arguments=raw_arguments,
    )


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
        after = stripped[len(FENCE_OPEN) :]
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
        blocks.append(
            json.dumps(
                {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                },
                indent=2,
            )
        )
    tool_list = "\n".join(blocks)

    # Import inside the function on purpose: agent_runtime is a pure module
    # today (no Textual, app, DB, or I/O imports per the module docstring)
    # and P1's import-hygiene philosophy keeps prompt plumbing out of module
    # import paths that don't need it. A module-level import would also
    # pass the hygiene test; this is the more conservative choice.
    from tldw_chatbook.Internal_Prompts import render_internal_prompt

    return render_internal_prompt(
        "agents.tool_protocol",
        tool_list=tool_list,
        fence_open=FENCE_OPEN,
        fence_close=_FENCE_CLOSE,
    )


@dataclass
class LoopDeps:
    """Everything impure, injected. The loop itself stays pure."""

    call_model: Callable[[list, tuple], ModelTurn]
    invoke_tool: Callable[..., ToolResult]
    # Callable[..., ...] (not Callable[[str], ...]): the loop itself only
    # ever calls spawn(task) positionally, or spawn(task, agent=...) when
    # the model supplied a named agent (fleet spec §4). The real
    # implementation (agent_service._run_one's spawn closure) also accepts
    # a keyword-only `allowed_tools` override, used by the skill-tool
    # dispatch path (SkillRunner.run) to narrow a spawned child's
    # allow-list -- the loop never passes THAT one and never needs to.
    spawn: Callable[..., ToolResult]
    find_tools: Callable[[str], list]
    load_schemas: Callable[[list], list]
    should_cancel: Callable[[], bool]
    clock: Callable[[], float]
    call_model_with_continuation: (
        Callable[
            [list, tuple, ProviderContinuationCheckpoint | None],
            ModelTurn,
        ]
        | None
    ) = None
    on_step: Callable[[AgentStep], None] = lambda step: None
    on_trace_step: Callable[[AgentStep], None] = lambda step: None
    # Optional pre-dispatch batch-review hook (P5 Task 4): the generic seam
    # the MCP approval flow (Task 6) rides on. When set, called ONCE per
    # turn with the full batch of tool calls about to be dispatched
    # (native multi-call batch, or the single fence-parsed call), BEFORE
    # any of them is invoked. Returns a name -> verdict map; "proceed"
    # (or an absent name -- a call the hook doesn't mention is presumed
    # fine) dispatches normally, anything else is treated as a refusal
    # string that is fed back to the model as that call's tool result
    # instead of invoking it. Exceptions fail closed for continuation
    # batches; legacy non-continuation batches retain their fail-open
    # behavior. ``None`` (the default) is a no-op: every call proceeds,
    # byte-identical to pre-Task-4 behavior.
    review_tool_calls: Callable[[list[ToolCall]], dict[str, str]] | None = None
    # Optional Task 10 whole-batch preparation. It runs once after the
    # assistant turn has entered run-local history and immediately before
    # the unchanged review hook. A retry result appends canonical deferral
    # stubs plus separate ephemeral context and returns to the model without
    # reviewing or dispatching any call in the deferred batch.
    prepare_tool_calls: Callable[[list[ToolCall]], ToolBatchPreparation] | None = None
    project_instruction_payload_state: InstructionChainPayloadState | None = None
    on_ephemeral_runtime_warning: Callable[[str, tuple[str, ...], int], None] | None = (
        None
    )
    # skill_file: the fourth runtime tool (task-3, skills-foundation). Unlike
    # a ToolProvider entry, its schema is pinned into runtime_schemas by the
    # service (never disclosure-gated) and its authorization lives on a
    # per-run SkillFileBindings object -- never config.allowed_tools. `None`
    # (the default) means the service never wired this run for skill_file at
    # all, and a call by that name falls through to the same
    # deps.invoke_tool path any other unrecognized/undisclosed name hits.
    read_skill_file: Callable[[str, str], ToolResult] | None = None
    # install_skill: the fifth runtime tool (agent-callable skill install).
    # Wired ONLY for the top-level agent (agent_kind == primary) by the
    # service; a spawned subagent never receives it. `None` (the default)
    # means the run is not wired for install_skill and a call by that name
    # falls through to the generic deps.invoke_tool path.
    install_skill: Callable[[str], ToolResult] | None = None
    # run_skill_script: the sixth runtime tool (trust-gated script execution).
    # Unlike install_skill this is NOT agent_kind-gated -- the user chose an
    # all-agents caller scope, because the per-run confirm card and the
    # per-skill grant (not the caller's identity) are what gate each run.
    # `None` (the default) means the run is not wired for it and a call by
    # that name falls through to the generic deps.invoke_tool path.
    run_skill_script: Callable[[str, str, list[str]], ToolResult] | None = None
    # search_run_log: the seventh runtime tool (run-log query). Wired ONLY
    # for the top-level agent (agent_kind == primary), like install_skill:
    # a depth-1 child has max_subagents clamped to 0, so its "subtree" is
    # itself and its short history is already in its context -- the tool
    # would buy it nothing while widening what it can see. `None` (the
    # default) means the run is not wired for it and a call by that name
    # falls through to the generic deps.invoke_tool path.
    search_run_log: Callable[[dict], ToolResult] | None = None
    # run_log_stats: Phase 2's aggregation runtime tool (design spec §10,
    # task-1271). Wired ONLY for the top-level agent, under the SAME gate
    # as search_run_log above and for the identical reason: a spawned
    # child's own short history is already in its context, so offering it
    # a tool that computes over the run TREE's shared log would only widen
    # what a child can see -- past its parent's history, contradicting
    # spawn_subagent's "sees only the task text" isolation promise. `None`
    # (the default) means the run is not wired for it and a call by that
    # name falls through to the generic deps.invoke_tool path.
    run_log_stats: Callable[[dict], ToolResult] | None = None
    # run_log_slice: Phase 2's contiguous-range retrieval runtime tool
    # (design spec §10, task-1271). Same primary-agent-only gate and
    # rationale as run_log_stats/search_run_log immediately above. `None`
    # (the default) means the run is not wired for it and a call by that
    # name falls through to the generic deps.invoke_tool path.
    run_log_slice: Callable[[dict], ToolResult] | None = None
    # wait_agents / check_agents: the fleet runtime tools (PR2a Task 6).
    # Wired ONLY for the top-level agent of a run that actually has a
    # fleet coordinator -- same primary-only reasoning as install_skill
    # (a depth-1 child has max_subagents clamped to 0, so it has no
    # children to wait on) plus the obvious one: without a coordinator
    # there is nothing to wait on at all. `None` (the default) means the
    # run is not wired for them and a call by either name falls through
    # to the generic deps.invoke_tool path.
    #
    # Dispatched IN-LOOP beside spawn_subagent below, deliberately NOT
    # through deps.invoke_tool: that path wraps every call in
    # agent_service._call_with_timeout's per-call daemon thread, which
    # would abandon a wait at `max_tool_call_seconds` and leave the
    # children it was waiting for running unattended. wait_agents is
    # bounded by the parent's own remaining wall-clock and polls
    # should_cancel itself, which is the correct bound for a call whose
    # entire purpose is to block.
    #
    # `wait_agents` takes the requested handle ids, or None for "every
    # child of this run"; `check_agents` takes nothing.
    wait_agents: Callable[[list[str] | None], ToolResult] | None = None
    check_agents: Callable[[], ToolResult] | None = None
    # send_to_agent: fleet steering, the SUPERVISOR producer (PR3b Task 2,
    # spec SS6) for the per-child mailbox drain_mailbox below consumes.
    # Wired under the exact `fleet_active` predicate as the two fields
    # above, same primary-only reasoning (depth-1: children cannot steer
    # each other). Takes (id, message) -- the id in either vocabulary,
    # resolved by the service closure -- and returns immediately: posting
    # to the locked in-memory mailbox never blocks, so dispatching it
    # in-loop beside wait_agents costs nothing and keeps all three fleet
    # tools on one path. Validation (non-empty, MAX_STEERING_CHARS) and
    # every piece of refusal copy live in the service closure, not here.
    send_to_agent: Callable[[str, str], ToolResult] | None = None
    # drain_mailbox: fleet steering (PR3b Task 1, spec SS6). Wired ONLY for
    # a THREADED fleet child -- the service's spawn tail closes it over
    # that child's own coordinator mailbox
    # (`FleetCoordinator.drain_steering`); primaries and inline children
    # get None (a primary is steered by the user talking to it; an inline
    # child has no handle and so no mailbox). Called by the loop at the
    # single protocol-coherent point -- in the non-restoring branch,
    # immediately before each model call, AFTER the budget/cancel checks --
    # returning the queued `(source, text)` entries, which the loop appends
    # as `format_steering_message`-labeled user-role messages. Draining
    # there can never split a native `tool_calls` <-> `role:"tool"` pair
    # (every batch's results are fully appended by then via
    # `_append_tool_result`), never collides with `expand_restore_history`'s
    # slice-rewrite (the restoring branch is structurally skipped), and a
    # dead run (cancelled/stuck/exhausted) never consumes a mailbox.
    # Wrapped never-raise at the call site, the `on_step` containment rule:
    # a broken drain costs the delivery, never the run.
    drain_mailbox: Callable[[], list[tuple[str, str]]] | None = None
    # on_record: full-fidelity capture for the run log (run_log.py). Called
    # with (record_type, payload) at the two points where the COMPLETE value
    # exists -- which the step log does not carry, since `add()` truncates
    # model turns to 200 chars and tool results to 2000. Captured in the
    # loop rather than in service wrappers because the loop assembles
    # `content` for EVERY dispatch branch at one point: a wrapper around
    # deps.invoke_tool would silently miss find_tools, load_tools,
    # spawn_subagent, skill_file, install_skill and run_skill_script.
    # `None` (the default) is a no-op: behavior is byte-identical to
    # pre-run-log runs.
    on_record: Callable[[str, dict], int | None] | None = None
    continuation_context: ContinuationEventContext | None = None
    persist_provider_continuation: Callable[[ProviderContinuationEvent], None] = (
        _noop_provider_continuation
    )
    expand_provider_continuation: (
        Callable[[ProviderContinuationCheckpoint], list[dict]] | None
    ) = None
    # Appended after every pre-existing field to preserve LoopDeps' legacy
    # positional constructor slots. Unlike ``clock`` (monotonic budgets),
    # this clock supplies UTC audit timestamps.
    wall_clock: Callable[[], datetime] = _utc_now


def _continuation_calls_match(
    checkpoint: ProviderContinuationCheckpoint,
    calls: list[ToolCall],
    assistant_content: str,
    assistant_message: dict | None,
) -> bool:
    """Return whether the active checkpoint's newest batch is exactly ``calls``."""
    try:
        dump_provider_continuation_json(checkpoint)
        canonical_calls = checkpoint.rounds[-1].calls
        echoed_content = (
            assistant_message.get("content") if assistant_message is not None else None
        )
        if (
            checkpoint.state != "active"
            or checkpoint.rounds[-1].assistant_content != assistant_content
            or echoed_content not in {None, assistant_content}
            or len(canonical_calls) != len(calls)
        ):
            return False
        if len({call.call_id for call in calls}) != len(calls):
            return False
        for canonical, call in zip(canonical_calls, calls):
            if (
                not call.call_id
                or not call.raw_arguments.strip()
                or canonical.call_id != call.call_id
                or canonical.name != call.name
                or canonical.state != "pending"
                or canonical.arguments != call.raw_arguments
            ):
                return False
    except Exception:
        return False
    return True


def _valid_continuation_context(
    context: ContinuationEventContext | None,
    persist: Callable[[ProviderContinuationEvent], None],
) -> bool:
    if (
        context is None
        or type(context.run_id) is not str
        or not context.run_id.strip()
        or context.agent_kind not in {"primary", "subagent", "fleet"}
        or context.durability not in {"persistent", "ephemeral"}
    ):
        return False
    if context.durability == "persistent":
        return bool(
            type(context.owner_message_id) is str
            and context.owner_message_id.strip()
            and persist is not _noop_provider_continuation
        )
    return True


def _same_continuation_target(
    current: ProviderContinuationCheckpoint,
    candidate: ProviderContinuationCheckpoint,
) -> bool:
    return (
        current.schema_version,
        current.provider,
        current.protocol,
        current.model,
        current.api_base_url,
    ) == (
        candidate.schema_version,
        candidate.provider,
        candidate.protocol,
        candidate.model,
        candidate.api_base_url,
    )


def _valid_batch_update(
    current: ProviderContinuationCheckpoint | None,
    candidate: ProviderContinuationCheckpoint,
) -> bool:
    if current is None:
        return candidate.checkpoint_revision == 1
    return (
        _same_continuation_target(current, candidate)
        and candidate.checkpoint_revision == current.checkpoint_revision + 1
        and candidate.rounds[:-1] == current.rounds
    )


def _valid_final_update(
    current: ProviderContinuationCheckpoint,
    candidate: ProviderContinuationCheckpoint,
    assistant_content: str,
) -> bool:
    if (
        candidate.state != "complete"
        or not _same_continuation_target(current, candidate)
        or candidate.checkpoint_revision != current.checkpoint_revision + 1
    ):
        return False
    if current.provider == "moonshot" and moonshot_model_returns_reasoning_content(
        current.model
    ):
        final_round = candidate.rounds[-1]
        if not final_round.calls:
            return (
                candidate.rounds[:-1] == current.rounds
                and final_round.assistant_content == assistant_content
            )
        # Reasoning-absent family completion keeps the pre-19170 durable
        # shape (rounds unchanged). kimi-k3 cannot reach this arm through
        # canonical parse -- its complete checkpoints must end with the
        # final reasoning round (TASK-19170).
        return candidate.rounds == current.rounds
    return candidate.rounds == current.rounds


def _catalog_lines(entries: list) -> str:
    if not entries:
        return "No matching tools."
    return "\n".join(f"{e.id} — {e.name}: {e.one_line_description}" for e in entries)


def _emit_record(deps: "LoopDeps", record_type: str, **payload) -> int | None:
    """Best-effort run-log capture; a failing writer never aborts a run.

    Args:
        deps: The run's injected dependencies.
        record_type: ``model``, ``tool_call``, or ``tool_result``.
        **payload: ``content``, ``tool``, ``status``, ``call_id``.

    Returns:
        The assigned record number, or ``None`` when logging is off or the
        write failed. Task 7 threads this into the truncation trailer.
    """
    if deps.on_record is None:
        return None
    try:
        return deps.on_record(record_type, payload)
    except Exception:  # noqa: BLE001 — logging is never load-bearing
        logger.opt(exception=True).warning(
            f"on_record hook raised for a {record_type} record; continuing"
        )
        return None


def _truncate_tool_result(
    content: str,
    max_chars: int,
    tool_name: str,
    record_number: int | None = None,
    *,
    total_limit: bool = False,
) -> str:
    """Bound one tool result before it enters history.

    Applied at the append seam rather than inside each tool so a tool that
    forgets to paginate cannot blow the context, and so MCP and skill
    results are covered by the same rule as built-ins.

    Args:
        content: The tool's full result text.
        max_chars: Ceiling from ``RunBudget.max_tool_result_chars``; 0 or
            negative means unlimited.
        tool_name: Named in the trailer so the model knows which call was
            cut and can re-issue it more narrowly.
        record_number: The run-log record number the untruncated result was
            captured under, or ``None`` when logging is off or the capture
            failed. When given, the trailer points at it via
            ``search_run_log`` instead of suggesting a re-issue. F7 (Qodo
            #7): may carry a truthy ``.truncated`` attribute (see
            ``run_log.RunLogRecordNumber``) reporting that the LOG ITSELF
            capped that record at ``run_log_max_record_bytes`` -- a plain
            ``int`` (every pre-existing caller, including every test that
            fabricates a bare record number) is read via ``getattr(...,
            "truncated", False)`` and always treated as "not capped", so
            this stays backward compatible.
        total_limit: Include the truncation trailer inside ``max_chars``.
            The default preserves the legacy contract where ``max_chars``
            limits the retained result prefix and the trailer is appended.

    Returns:
        ``content`` unchanged when under the cap or when unlimited. Otherwise,
        the legacy default returns the first ``max_chars`` characters plus a
        trailer; ``total_limit=True`` bounds the complete returned string.
    """
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    if record_number is not None:
        # F7 (Qodo #7): a record the WRITER itself had to cap (content over
        # run_log_max_record_bytes, 1MB default) has an unrecoverable tail
        # of its own -- pointing at it as "the full result" would be a
        # second, compounding false promise on top of this history cut.
        # `getattr` defaults to False so a plain int (every record number
        # before this fix, and every record that was NOT capped) takes the
        # unconditional-recovery wording unchanged.
        if getattr(record_number, "truncated", False):
            recovery = (
                f" Record {record_number:06d} holds as much as this run's "
                f"log could store under its own per-record cap -- the "
                f"remainder was never written and cannot be recovered. "
                f"search_run_log(from_record={record_number}, "
                f"to_record={record_number}) shows exactly how much was kept."
            )
        else:
            # TASK-1250: a bare from_record/to_record call renders the SAME
            # first `max_chars` this trailer already cut -- format_results
            # windows at this run's own tool-result ceiling, so it cannot
            # show more in one call. Naming `contains=`/`offset=` here is
            # what makes the pointer actually deliver content beyond this
            # cut, instead of promising recovery a bare call can't provide.
            recovery = (
                f" The full result is recorded at record {record_number:06d} — "
                f"search_run_log(from_record={record_number}, to_record={record_number}) "
                f"renders it windowed at this same limit, so add contains=<term> to "
                f"jump straight to a match, or offset=<n> to page past it (the "
                f"rendered output states the next offset)."
            )
    else:
        recovery = (
            " Re-issue the call with a narrower query, or use the tool's "
            "offset/limit arguments to read the rest."
        )
    full_trailer = (
        f"\n\n[truncated: {tool_name} returned {len(content)} characters.{recovery}]"
    )
    if not total_limit:
        return content[:max_chars] + full_trailer
    trailer = next(
        (
            value
            for value in (
                full_trailer,
                f"\n[truncated: {tool_name} {len(content)}]",
                "\n[truncated]",
                "[cut]",
            )
            if len(value) < max_chars
        ),
        "",
    )
    if not trailer:
        return content[:max_chars]
    return content[: max_chars - len(trailer)] + trailer


def _append_tool_result(messages: list[dict], call: ToolCall, content: str) -> None:
    """Append one tool result to history using the call's role/id shaping.

    Single source of truth for both the normal post-invoke path and the
    review-hook refusal path (P5 Task 4) so the two can never drift.

    Native protocol (``call.call_id`` set): a ``role="tool"`` message
    paired to the assistant turn's ``tool_calls`` entry by
    ``tool_call_id``. Fence protocol (``call.call_id`` unset): the
    plain-text ``"{FENCE_TOOL_RESULT_PREFIX}{name}: {content}"``
    convention, appended as a user-role message. ``FENCE_TOOL_RESULT_
    PREFIX`` is a shared constant (``agent_models``) so
    ``run_log_eviction``'s protocol-aware turn grouping matches this exact
    string rather than a copy that could drift from it.
    """
    if call.call_id:
        messages.append(
            {"role": "tool", "tool_call_id": call.call_id, "content": content}
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": f"{FENCE_TOOL_RESULT_PREFIX}{call.name}: {content}",
            }
        )


def _detect_cycle(recent) -> tuple[int, int] | None:
    """Detect a repeating tool-call cycle in the tail of ``recent``.

    Returns ``(period, repeats)`` when the last ``repeats*period`` call-keys
    are ``repeats`` consecutive copies of the trailing ``period``-block, else
    ``None``. Threshold: ``LOOP_DETECTION_N`` (3) repeats for period 1
    (backward-compatible with the prior identical-consecutive check), 2 for
    periods >= 2. Smallest period first, so a longer cycle is never
    mis-attributed to a shorter period. Pure (no I/O).
    """
    seq = list(recent)
    n = len(seq)
    for period in range(1, MAX_LOOP_PERIOD + 1):
        repeats = LOOP_DETECTION_N if period == 1 else 2
        need = repeats * period
        if n < need:
            continue
        tail = seq[-need:]
        block = tail[-period:]
        if all(tail[i] == block[i % period] for i in range(need)):
            return (period, repeats)
    return None


def run_agent_loop(
    config: AgentConfig,
    initial_messages: list[dict],
    active_schemas: list,
    deps: LoopDeps,
    *,
    restore_provider_continuation: ProviderContinuationCheckpoint | None = None,
    restore_provider_target: ContinuationRestoreTarget | None = None,
    resume_provider_continuation: bool = False,
) -> RunOutcome:
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
            provider-call/model-turn count all independently cap the run;
            task-326 adds ``max_total_tokens``, a cumulative prompt+
            completion token spend ceiling — 0 means unlimited).
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
        answer text (when done), how many sub-agents were spawned, and
        (task-326) ``total_tokens`` — the measured cumulative prompt+
        completion token spend checked against ``max_total_tokens``.
    """
    budget = config.budget
    steps: list[AgentStep] = []
    messages = list(initial_messages)
    # PR3b Task 4: how much of `messages` is protocol-coherent -- i.e. up
    # to the LAST drain boundary (the pre-model-call point where every
    # previous batch's results are fully appended). Every terminal return
    # captures `messages[:coherent_len]` onto the outcome as
    # `final_messages`, so a mid-batch death (cancel `should_cancel`
    # inside the `for call in calls:` body, cycle-stuck) slices the whole
    # split batch away instead of retaining a half-answered native
    # `tool_calls` pair. Initialized to the seed history so the pre-loop
    # terminal paths (restore validation failures) are covered too.
    coherent_len = len(messages)
    active = list(active_schemas)
    started = deps.clock()
    spawned = 0
    model_turns = 0
    total_tokens = 0
    budget_steps = 0
    trace_steps = 0
    continuation_checkpoint: ProviderContinuationCheckpoint | None = None
    restored_calls: list[ToolCall] | None = None
    restore_history_start: int | None = None
    recent_calls: deque = deque(maxlen=LOOP_DETECTION_N * MAX_LOOP_PERIOD)

    def add(kind: str, *, counts_toward_budget: bool = True, **kw) -> AgentStep:
        nonlocal budget_steps
        if not kw.get("created_at"):
            kw["created_at"] = safe_utc_timestamp(deps.wall_clock)
        step = AgentStep(index=len(steps), kind=kind, **kw)
        steps.append(step)
        if counts_toward_budget:
            budget_steps += 1
        # The service composes incremental durability with live UI here.
        # Either callback may fail, but observation is never load-bearing
        # for the run itself.
        try:
            deps.on_step(step)
        except Exception:  # noqa: BLE001 — best-effort observation only
            pass
        return step

    def trace(kind: str, **kw) -> None:
        """Capture a safe lifecycle observation outside legacy control steps."""
        nonlocal trace_steps
        if not kw.get("created_at"):
            kw["created_at"] = safe_utc_timestamp(deps.wall_clock)
        step = AgentStep(index=1_000_000 + trace_steps, kind=kind, **kw)
        trace_steps += 1
        try:
            deps.on_trace_step(step)
        except Exception:  # noqa: BLE001 — best-effort observation only
            pass

    def _outcome(status: str, **kw) -> RunOutcome:
        # Reports run spend on every terminal path; reads enclosing steps/
        # spawned/total_tokens at call time (no nonlocal, like add()).
        #
        # PR3b Task 4: every terminal return also captures the coherent
        # transcript -- `messages[:coherent_len]`, the last drain-boundary
        # prefix. RUN_DONE additionally appends the final assistant text:
        # the done-return fires BEFORE the loop's own assistant append
        # (see the `if not calls:` return below), so without this the
        # retained transcript would end on the user turn the model just
        # answered. Only RUN_DONE gets it -- a cancel that lands mid-way
        # through a streamed final turn (`final_text` set on
        # RUN_CANCELLED) never delivered that text as a completed turn,
        # and pretending otherwise would fabricate the child's memory.
        final_messages = [
            row
            for row in messages[:coherent_len]
            if row.get(EPHEMERAL_ORIGIN_KEY) != "project_instructions"
        ]
        if status == RUN_DONE:
            final_messages = final_messages + [
                {"role": "assistant", "content": kw.get("final_text", "")}
            ]
        return RunOutcome(
            status,
            steps,
            subagents_spawned=spawned,
            total_tokens=total_tokens,
            final_messages=final_messages,
            **kw,
        )

    def continuation_error() -> RunOutcome:
        add(
            STEP_ERROR,
            summary=(
                "Provider continuation could not be persisted; retry or recover "
                "the interrupted run."
            ),
        )
        return _outcome(RUN_ERROR)

    def persist_continuation(event: ProviderContinuationEvent) -> bool:
        try:
            deps.persist_provider_continuation(event)
        except Exception:
            return False
        return True

    def transition_call(
        call: ToolCall,
        target: str,
        result: ContinuationResult | None = None,
    ) -> bool:
        nonlocal continuation_checkpoint
        checkpoint = continuation_checkpoint
        context = deps.continuation_context
        if checkpoint is None or context is None:
            return False
        revision = checkpoint.checkpoint_revision
        if target == "executing":
            event: ProviderContinuationEvent = ToolCallExecuting(
                context=context,
                call_id=call.call_id,
                expected_checkpoint_revision=revision,
            )
        else:
            if result is None:
                return False
            event = ToolCallFinished(
                context=context,
                call_id=call.call_id,
                expected_checkpoint_revision=revision,
                target_state=target,  # type: ignore[arg-type]
                result=result,
            )
        if not persist_continuation(event):
            return False
        try:
            continuation_checkpoint = transition_provider_call(
                checkpoint,
                call_id=call.call_id,
                expected_revision=revision,
                target=target,  # type: ignore[arg-type]
                result=result,
            )
        except Exception:
            return False
        return True

    def expand_restore_history(checkpoint: ProviderContinuationCheckpoint) -> bool:
        nonlocal restore_history_start
        expand = deps.expand_provider_continuation
        if expand is None:
            return False
        try:
            rows = expand(checkpoint)
        except Exception:
            return False
        if type(rows) is not list or any(type(row) is not dict for row in rows):
            return False
        if restore_history_start is None:
            restore_history_start = len(messages)
        messages[restore_history_start:] = rows
        return True

    if restore_provider_continuation is not None:
        context = deps.continuation_context
        checkpoint = restore_provider_continuation
        try:
            if restore_provider_target is None:
                raise ValueError
            validate_continuation_restore(checkpoint, restore_provider_target)
        except Exception:
            return continuation_error()
        if checkpoint.state != "active" or not _valid_continuation_context(
            context, deps.persist_provider_continuation
        ):
            return continuation_error()
        if not resume_provider_continuation:
            add(
                STEP_ERROR,
                summary="Provider continuation is paused; explicit resume is required.",
            )
            return _outcome(RUN_STUCK)
        if any(
            call.state == "executing"
            for round_ in checkpoint.rounds
            for call in round_.calls
        ):
            add(
                STEP_ERROR,
                summary=(
                    "Provider continuation is blocked because a tool result is "
                    "ambiguous; discard and retry safely."
                ),
            )
            return _outcome(RUN_STUCK)
        if not expand_restore_history(checkpoint):
            return continuation_error()
        continuation_checkpoint = checkpoint
        restored_calls = []
        for round_ in checkpoint.rounds:
            calls = [
                ToolCall(
                    name=call.name,
                    args=json.loads(call.arguments),
                    call_id=call.call_id,
                    raw_arguments=call.arguments,
                )
                for call in round_.calls
            ]
            for canonical, call in zip(round_.calls, calls):
                if canonical.state == "pending":
                    restored_calls.append(call)

    while True:
        if deps.should_cancel():
            trace(
                STEP_MODEL_CANCELLED,
                summary="Model request cancelled",
                status="cancelled",
                field_states={"payload": "omitted"},
                sensitivity="diagnostic",
            )
            return _outcome(RUN_CANCELLED)
        if budget_steps >= budget.max_steps:
            add(STEP_ERROR, summary="step budget exhausted")
            return _outcome(RUN_STUCK)
        if model_turns >= budget.max_model_turns:
            add(STEP_ERROR, summary="model-turn budget exhausted")
            return _outcome(RUN_STUCK)
        if deps.clock() - started > budget.max_wall_seconds:
            add(STEP_ERROR, summary="wall-clock budget exhausted")
            return _outcome(RUN_STUCK)
        if budget.max_total_tokens and total_tokens >= budget.max_total_tokens:
            add(STEP_ERROR, summary="token budget exhausted")
            return _outcome(RUN_STUCK)

        restoring_batch = restored_calls is not None and bool(restored_calls)
        ephemeral_continuation = False
        if restoring_batch:
            calls = restored_calls or []
            restored_calls = None
            turn = ModelTurn(tool_calls=tuple(calls))
        else:
            restored_calls = None
            # Fleet steering drain (PR3b Task 1) -- THE protocol-coherent
            # point, deliberately here and nowhere else: every previous
            # batch's results are fully appended by now (via
            # `_append_tool_result`, both protocols), the restoring branch
            # above is structurally skipped (so this can never collide
            # with `expand_restore_history`'s slice-rewrite), and the
            # budget/cancel checks at the loop top ran first (a dead run
            # never consumes a mailbox). Wrapped never-raise like
            # `on_step`: a broken drain costs the delivery, never the run.
            if deps.drain_mailbox is not None:
                try:
                    for steer_source, steer_text in deps.drain_mailbox():
                        steer_message = format_steering_message(
                            steer_source, steer_text
                        )
                        messages.append({"role": "user", "content": steer_message})
                        add(STEP_STEERING, summary=steer_message[:200])
                        _emit_record(
                            deps,
                            "steering",
                            content=steer_message,
                            tool="",
                            status=steer_source,
                            call_id="",
                        )
                except Exception:  # noqa: BLE001 — containment, like on_step
                    logger.opt(exception=True).warning(
                        "drain_mailbox raised; steering delivery skipped for this turn"
                    )
            # PR3b Task 4: THE boundary. Everything appended so far --
            # every previous batch's fully-paired results, and the
            # steering messages just delivered above -- is exactly what
            # this model call is about to see, so this is the largest
            # prefix a retained transcript may ever carry. Set AFTER the
            # drain so delivered steering is part of what a resumed child
            # remembers (it saw it), and never updated in the restoring
            # branch (whose slice-rewrite makes any mid-restore capture
            # unsafe -- a terminal return there keeps the previous
            # boundary instead).
            coherent_len = len(messages)
            trace(
                STEP_MODEL_REQUEST_STARTED,
                summary="Model request started",
                status="started",
                field_states={"payload": "omitted"},
                sensitivity="diagnostic",
            )
            try:
                turn = (
                    deps.call_model_with_continuation(
                        messages,
                        tuple(active),
                        continuation_checkpoint,
                    )
                    if deps.call_model_with_continuation is not None
                    else deps.call_model(messages, tuple(active))
                )
            except Exception:
                trace(
                    STEP_MODEL_ERROR,
                    summary="Model request failed",
                    status="failed",
                    field_states={"payload": "omitted"},
                    sensitivity="diagnostic",
                )
                raise
            trace(
                STEP_MODEL_RESPONSE_COMPLETED,
                summary="Model response completed",
                status="completed",
                field_states={"payload": "omitted"},
                sensitivity="diagnostic",
            )
            model_turns += 1
            total_tokens += turn.tokens
            calls = list(turn.tool_calls)
        fenced = None
        if not calls:
            _visible, fenced = split_visible_text_and_tool_call(turn.text)
            if fenced is not None:
                calls = [fenced]
        candidate = turn.provider_continuation
        if not restoring_batch and calls and candidate is not None:
            context = deps.continuation_context
            if not _valid_continuation_context(
                context, deps.persist_provider_continuation
            ) or not _continuation_calls_match(
                candidate, calls, turn.text, turn.assistant_message
            ):
                return continuation_error()
            expected_revision = (
                continuation_checkpoint.checkpoint_revision
                if continuation_checkpoint is not None
                else None
            )
            if not _valid_batch_update(continuation_checkpoint, candidate):
                return continuation_error()
            batch_event = ToolBatchReady(
                context=context,  # type: ignore[arg-type]
                checkpoint=candidate,
                expected_checkpoint_revision=expected_revision,
            )
            if not persist_continuation(batch_event):
                return continuation_error()
            continuation_checkpoint = candidate
            ephemeral_continuation = bool(
                context is not None and context.durability == "ephemeral"
            )
        elif not restoring_batch and calls and continuation_checkpoint is not None:
            return continuation_error()

        if not calls:
            # A Stop can land while this (tool-call-free) turn was still
            # streaming. There is no further step/tool-call boundary ahead.
            if deps.should_cancel():
                return _outcome(RUN_CANCELLED, final_text=turn.text)
            if candidate is not None:
                context = deps.continuation_context
                try:
                    dump_provider_continuation_json(candidate)
                except Exception:
                    return continuation_error()
                if (
                    not _valid_continuation_context(
                        context, deps.persist_provider_continuation
                    )
                    or candidate.state != "complete"
                ):
                    return continuation_error()
                expected_revision = (
                    continuation_checkpoint.checkpoint_revision
                    if continuation_checkpoint is not None
                    else None
                )
                if continuation_checkpoint is None:
                    final_round = candidate.rounds[-1]
                    if (
                        candidate.checkpoint_revision != 1
                        or candidate.provider != "moonshot"
                        or not moonshot_model_returns_reasoning_content(candidate.model)
                        or len(candidate.rounds) != 1
                        or final_round.calls
                        or final_round.assistant_content != turn.text
                    ):
                        return continuation_error()
                elif not _valid_final_update(
                    continuation_checkpoint, candidate, turn.text
                ):
                    return continuation_error()
                final_event = FinalContinuation(
                    context=context,  # type: ignore[arg-type]
                    checkpoint=candidate,
                    expected_checkpoint_revision=expected_revision,
                    assistant_content=turn.text,
                )
                if not persist_continuation(final_event):
                    return continuation_error()
                continuation_checkpoint = candidate
            elif continuation_checkpoint is not None:
                return continuation_error()

        if not restoring_batch:
            add(
                STEP_MODEL,
                summary=(
                    "Ephemeral tool continuation is non-resumable."
                    if ephemeral_continuation
                    else turn.text[:200]
                ),
            )
            _emit_record(
                deps,
                "model",
                content=turn.text,
                tool="",
                status="",
                call_id="",
            )
        if not calls:
            return _outcome(RUN_DONE, final_text=turn.text)
        if not restoring_batch:
            messages.append(
                turn.assistant_message or {"role": "assistant", "content": turn.text}
            )

        if deps.prepare_tool_calls is not None and calls:
            preparation = ToolBatchPreparation("proceed")
            try:
                if deps.project_instruction_payload_state is not None:
                    deps.project_instruction_payload_state.capture(
                        messages, tuple(active), calls
                    )
                preparation = deps.prepare_tool_calls(list(calls))
                if not isinstance(preparation, ToolBatchPreparation):
                    raise TypeError("invalid tool batch preparation")
            except Exception:  # noqa: BLE001 - content-free fail-open boundary
                code = "project_instruction_preparation_failed"
                tool_names = tuple(call.name for call in calls)
                logger.warning(
                    f"{code}: tool_names={tool_names!r} tool_count={len(calls)}"
                )
                if deps.on_ephemeral_runtime_warning is not None:
                    try:
                        deps.on_ephemeral_runtime_warning(code, tool_names, len(calls))
                    except Exception:  # noqa: BLE001 - warning is best effort
                        logger.warning("project_instruction_warning_callback_failed")
            if preparation.status == "retry_with_context":
                trace(
                    STEP_MODEL_RETRY,
                    summary="Model retry requested",
                    status="retrying",
                    field_states={"payload": "omitted"},
                    sensitivity="diagnostic",
                )
                messages.extend(build_project_instruction_deferral_rows(calls))
                messages.extend(
                    deepcopy(dict(row)) for row in preparation.ephemeral_rows
                )
                continue

        # P5 Task 4: optional pre-dispatch batch review, called ONCE with
        # the full batch about to be dispatched below (whichever produced
        # it -- native multi-call or the single fence-parsed call) and
        # BEFORE any of them is invoked. `deps.review_tool_calls is None`
        # (the default) short-circuits to an empty verdicts map, which
        # makes every `.get(name, "proceed")` lookup below resolve to
        # "proceed" -- the exact same dispatch path as before this hook
        # existed, so absent-hook behavior stays byte-identical.
        for call in calls:
            trace(
                STEP_TOOL_PROPOSED,
                summary=f"{call.name} proposed",
                tool_name=call.name,
                status="proposed",
                field_states={"args": "omitted", "result": "not_available"},
                sensitivity="tool_content",
            )

        verdicts: dict[str, str] = {}
        if deps.review_tool_calls is not None and calls:
            for call in calls:
                trace(
                    STEP_APPROVAL_REQUESTED,
                    summary=f"Approval requested for {call.name}",
                    tool_name=call.name,
                    status="pending",
                    field_states={"args": "omitted", "result": "not_available"},
                    sensitivity="tool_content",
                )
            try:
                verdicts = deps.review_tool_calls(list(calls)) or {}
            except Exception:  # noqa: BLE001 — policy differs by lifecycle
                if continuation_checkpoint is not None:
                    return continuation_error()
                # MCP-specific fail-closed policy lives in the Task 6
                # closure that builds this callable, not in this generic
                # runtime.
                logger.opt(exception=True).warning(
                    f"review_tool_calls hook raised for batch "
                    f"{[c.name for c in calls]}; treating all {len(calls)} "
                    f"calls as proceed"
                )
                verdicts = {}

            for call in calls:
                verdict = (
                    verdicts.get(call.call_id) if call.call_id else None
                ) or verdicts.get(call.name, "proceed")
                trace(
                    STEP_APPROVAL_APPROVED
                    if verdict == "proceed"
                    else STEP_APPROVAL_DENIED,
                    summary=(
                        f"Approval granted for {call.name}"
                        if verdict == "proceed"
                        else f"Approval denied for {call.name}"
                    ),
                    tool_name=call.name,
                    status="approved" if verdict == "proceed" else "denied",
                    field_states={"args": "omitted", "result": "omitted"},
                    sensitivity="tool_content",
                )

        for call in calls:
            # F5 (Qodo #5, PR #1066 review): emit the tool_call record BEFORE
            # the dispatch chain below, not after. `call.name`/`call.args`
            # are already known here, so nothing is gained by waiting -- and
            # waiting is exactly the bug: the old placement sat at the
            # content-assembly point, which runs AFTER the tool has already
            # executed (including, for SPAWN_TOOL_NAME, after `deps.spawn`
            # has run the ENTIRE child loop inline). A crash, a kill, or an
            # indefinitely blocked tool left no durable record the call was
            # ever attempted, and a child's own records -- written during
            # the parent's still-in-progress spawn dispatch -- landed in the
            # log BEFORE the parent's own record that caused them.
            # The continuation refusal path below emits this record only
            # after its atomic Finished barrier. All dispatch paths emit it
            # here, before the side effect, and legacy refusals retain their
            # pre-existing record order.
            if deps.should_cancel():
                if deps.review_tool_calls is not None:
                    trace(
                        STEP_APPROVAL_REVOKED,
                        summary=f"Approval revoked for {call.name}",
                        tool_name=call.name,
                        status="revoked",
                        field_states={"args": "omitted", "result": "omitted"},
                        sensitivity="tool_content",
                    )
                trace(
                    STEP_TOOL_CANCELLED,
                    summary=f"{call.name} cancelled",
                    tool_name=call.name,
                    status="cancelled",
                    field_states={"args": "omitted", "result": "not_available"},
                    sensitivity="tool_content",
                )
                return _outcome(RUN_CANCELLED)
            recent_calls.append((call.name, json.dumps(call.args, sort_keys=True)))
            cycle = _detect_cycle(recent_calls)
            if cycle is not None:
                period, repeats = cycle
                # Name the offending tool(s) so the user-facing "Agent run
                # stuck: ..." copy (console_chat_controller's
                # _agent_failure_visible_copy, which surfaces this summary
                # verbatim) stays actionable instead of reading as bare
                # "N-cycle" jargon. dict.fromkeys de-dupes while preserving
                # order (a period-1 trip names the tool once, not 3x).
                names = ", ".join(
                    dict.fromkeys(n for n, _ in list(recent_calls)[-period * repeats :])
                )
                # Log-side detail (TASK-1231/F3 AC4): the period/repeats
                # jargon that used to be the ONLY copy this trip produced is
                # kept here, at debug level, for anyone actually debugging
                # the cycle detector -- but it must never be the user-facing
                # `summary` below, which console_chat_controller surfaces
                # verbatim as "Agent run stuck: {summary}." Fleet-UX review
                # F3: "loop detected: read_file repeated in a 1-cycle (3x)"
                # reads as unexplained jargon to a first-run user.
                logger.debug(
                    f"loop detected: period={period} repeats={repeats} tools={names}"
                )
                if period == 1:
                    summary = (
                        f"Agent stopped: it kept calling {names} with the "
                        f"same arguments ({repeats} times) without making "
                        "progress."
                    )
                else:
                    summary = (
                        f"Agent stopped: it kept repeating the same "
                        f"sequence of tool calls ({names}) without making "
                        "progress."
                    )
                add(STEP_ERROR, summary=summary)
                return _outcome(RUN_STUCK)

            # P5 Task 4: a non-"proceed" verdict (an absent name defaults to
            # "proceed" — the hook only reports what it wants to stop)
            # skips dispatch entirely: none of the SPAWN/find_tools/
            # load_tools/invoke_tool branches below run, and the verdict
            # string itself becomes the call's tool result, same as any
            # other result content from here down.
            # Verdict lookup is PER CALL first, then by name.
            #
            # It used to be name-only, which meant same-name calls in one
            # batch shared one verdict: a turn reading two files was a single
            # yes/no, so you could not allow `spec.md` and refuse
            # `secrets.md`. Tools are how an agent reaches the outside world,
            # so per-target granularity is the point of the gate.
            #
            # The name fallback is load-bearing, not legacy politeness:
            # `MCPToolProvider.apply_batch_decisions` emits name-keyed
            # verdicts, and the fence path builds ToolCalls with NO call_id
            # at all (`parse_tool_call`), so a name-keyed verdict must still stop
            # every matching call or the MCP gate silently opens.
            verdict = "proceed"
            if call.call_id and call.call_id in verdicts:
                verdict = verdicts[call.call_id]
            else:
                verdict = verdicts.get(call.name, "proceed")
            if continuation_checkpoint is not None and verdict != "proceed":
                continuation_cap = (
                    min(budget.max_tool_result_chars, 16_000)
                    if budget.max_tool_result_chars > 0
                    else 16_000
                )
                content = _truncate_tool_result(
                    verdict,
                    continuation_cap,
                    call.name,
                    total_limit=True,
                )
                if not transition_call(call, "failed", ContinuationResult(content)):
                    return continuation_error()
                _emit_record(
                    deps,
                    "tool_call",
                    content=json.dumps(call.args, sort_keys=True, default=str),
                    tool=call.name,
                    status="",
                    call_id=call.call_id,
                )
                _emit_record(
                    deps,
                    "tool_result",
                    content=verdict,
                    tool=call.name,
                    status="refused",
                    call_id=call.call_id,
                )
                add(
                    STEP_TOOL_RESULT,
                    tool_name=call.name,
                    result=content[:2000],
                    tool_outcome=TOOL_OUTCOME_BLOCKED,
                )
                if restoring_batch:
                    if not expand_restore_history(continuation_checkpoint):
                        return continuation_error()
                else:
                    _append_tool_result(messages, call, content)
                continue
            if continuation_checkpoint is not None and not transition_call(
                call, "executing"
            ):
                return continuation_error()
            _emit_record(
                deps,
                "tool_call",
                content=json.dumps(call.args, sort_keys=True, default=str),
                tool=call.name,
                status="",
                call_id=call.call_id,
            )
            if verdict != "proceed":
                content = verdict
                tool_outcome = TOOL_OUTCOME_BLOCKED
            else:
                trace(
                    STEP_TOOL_EXECUTION_STARTED,
                    summary=f"{call.name} execution started",
                    tool_name=call.name,
                    status="started",
                    field_states={"args": "omitted", "result": "not_available"},
                    sensitivity="tool_content",
                )
                if call.name == SPAWN_TOOL_NAME:
                    if SPAWN_TOOL_NAME not in config.allowed_tools:
                        # Q6: refuse before dispatch — no budget consumption,
                        # no STEP_SPAWN, deps.spawn never called.
                        result = ToolResult(
                            ok=False, error=f"Tool not permitted: {SPAWN_TOOL_NAME}"
                        )
                    else:
                        task = str(call.args.get("task", "")).strip()
                        # `.get("agent") or ""` (not `.get("agent", "")`):
                        # an explicit JSON `null` for "agent" arrives here as
                        # Python `None`, and `str(None)` is the truthy
                        # string "None" -- which would then fail unknown-
                        # agent resolution with a spurious "unknown agent
                        # 'None'" refusal instead of taking the no-agent
                        # path.
                        agent_name = str(call.args.get("agent") or "").strip()
                        if not task:
                            # G4: an empty task is refused with no budget
                            # consumption and no STEP_SPAWN.
                            result = ToolResult(
                                ok=False, error="Task description cannot be empty"
                            )
                        elif spawned >= budget.max_subagents:
                            result = ToolResult(
                                ok=False, error="sub-agent budget exhausted"
                            )
                        else:
                            add(
                                STEP_SPAWN,
                                summary=(
                                    f"[{agent_name}] {task}"[:200]
                                    if agent_name
                                    else task[:200]
                                ),
                                tool_name=SPAWN_TOOL_NAME,
                                args=dict(call.args),
                            )
                            if agent_name:
                                result = deps.spawn(task, agent=agent_name)
                            else:
                                result = deps.spawn(task)
                            # Named-agent resolution (fleet spec §4) gave
                            # deps.spawn a NEW failure mode this loop-level
                            # check does not pre-screen: deps.spawn can now
                            # refuse a NAMED spawn for an unknown `agent`
                            # (or its own budget check) after this branch
                            # was already entered. This is a redundant
                            # secondary bound; the service's own
                            # sub_agent_spawns counter remains authoritative.
                            #
                            # Increment accounting differs by path:
                            # - No-agent path: increment is unconditional,
                            #   byte-identical to pre-task-5 behavior, including
                            #   spawns whose child ran and ended non-DONE.
                            # - Named path: increment only when result.ok.
                            #   Any named-spawn failure (unknown agent, budget
                            #   refusal before dispatch, or child ending
                            #   non-DONE) skips the counter; otherwise a later
                            #   VALID named spawn would be wrongly refused here
                            #   before ever reaching deps.spawn's own (real)
                            #   budget check.
                            if result.ok or not agent_name:
                                spawned += 1
                elif (
                    call.name == WAIT_AGENTS_TOOL_NAME and deps.wait_agents is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    # Same defensive coercion as load_tools' `ids` right
                    # below: an unreliable local model may send one bare
                    # string, a JSON null, or junk. A bare string is ONE
                    # id (never char-split); anything unusable becomes
                    # None, which the service reads as "every child" --
                    # the same thing an omitted `ids` means, and the
                    # safest reading of an ambiguous request to wait.
                    raw_ids = call.args.get("ids")
                    if isinstance(raw_ids, str):
                        wait_ids: list[str] | None = [raw_ids]
                    elif isinstance(raw_ids, list):
                        # An explicitly EMPTY list is "no ids given",
                        # i.e. all of them -- not "wait for nothing".
                        wait_ids = [str(x) for x in raw_ids] or None
                    else:
                        wait_ids = None
                    result = deps.wait_agents(wait_ids)
                elif (
                    call.name == CHECK_AGENTS_TOOL_NAME
                    and deps.check_agents is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.check_agents()
                elif (
                    call.name == SEND_TO_AGENT_TOOL_NAME
                    and deps.send_to_agent is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    # Same defensive coercion as wait_agents' `ids` above:
                    # an unreliable local model may send numbers or JSON
                    # nulls. Anything non-string becomes its str() form (a
                    # numeric id then simply fails to resolve, and the
                    # service's own refusal copy names it); a missing/null
                    # value becomes "" so the service's empty-message and
                    # unknown-id refusals speak, never a crash here.
                    raw_target = call.args.get("id")
                    raw_message = call.args.get("message")
                    result = deps.send_to_agent(
                        "" if raw_target is None else str(raw_target),
                        "" if raw_message is None else str(raw_message),
                    )
                elif call.name == FIND_TOOLS_NAME:
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    entries = deps.find_tools(str(call.args.get("query", "")))
                    result = ToolResult(ok=True, content=_catalog_lines(entries))
                elif call.name == LOAD_TOOLS_NAME:
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
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
                            ok=False, error="No valid tools found to load"
                        )
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
                        already_active = [
                            s.name for s in loaded if s.name in active_names
                        ]
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
                                content="already loaded: " + ", ".join(already_active),
                            )
                        else:
                            room = budget.max_active_tools - len(active)
                            accepted = new_loaded[: max(room, 0)]
                            active.extend(accepted)
                            if accepted:
                                result = ToolResult(
                                    ok=True,
                                    content="loaded: "
                                    + ", ".join(s.name for s in accepted),
                                )
                            else:
                                result = ToolResult(ok=True, content="no room")
                elif (
                    call.name == SKILL_FILE_TOOL_NAME
                    and deps.read_skill_file is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.read_skill_file(
                        str(call.args.get("skill_name", "")),
                        str(call.args.get("path", "")),
                    )
                elif (
                    call.name == INSTALL_SKILL_TOOL_NAME
                    and deps.install_skill is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.install_skill(str(call.args.get("url", "")))
                elif (
                    call.name == RUN_SKILL_SCRIPT_TOOL_NAME
                    and deps.run_skill_script is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    raw_args = call.args.get("args") or []
                    if not isinstance(raw_args, (list, tuple)):
                        raw_args = [raw_args]
                    result = deps.run_skill_script(
                        str(call.args.get("skill_name", "")),
                        str(call.args.get("script_path", "")),
                        [str(item) for item in raw_args],
                    )
                elif (
                    call.name == SEARCH_RUN_LOG_TOOL_NAME
                    and deps.search_run_log is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.search_run_log(dict(call.args))
                elif (
                    call.name == RUN_LOG_STATS_TOOL_NAME
                    and deps.run_log_stats is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.run_log_stats(dict(call.args))
                elif (
                    call.name == RUN_LOG_SLICE_TOOL_NAME
                    and deps.run_log_slice is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.run_log_slice(dict(call.args))
                else:
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.invoke_tool(call)

                tool_outcome = (
                    TOOL_OUTCOME_SUCCESS
                    if result.ok
                    else (
                        TOOL_OUTCOME_BLOCKED
                        if result.outcome == TOOL_OUTCOME_BLOCKED
                        else (
                            result.outcome
                            if result.outcome
                            in {TOOL_OUTCOME_TIMEOUT, TOOL_OUTCOME_CANCELLED}
                            else TOOL_OUTCOME_FAILED
                        )
                    )
                )
                content = result.content if result.ok else f"ERROR: {result.error}"

            terminal_kind = {
                TOOL_OUTCOME_SUCCESS: STEP_TOOL_SUCCEEDED,
                TOOL_OUTCOME_TIMEOUT: STEP_TOOL_TIMED_OUT,
                TOOL_OUTCOME_CANCELLED: STEP_TOOL_CANCELLED,
            }.get(tool_outcome, STEP_TOOL_FAILED)
            trace(
                terminal_kind,
                summary=f"{call.name} {tool_outcome}",
                tool_name=call.name,
                tool_outcome=tool_outcome,
                status=tool_outcome,
                field_states={"args": "omitted", "result": "omitted"},
                sensitivity="tool_content",
            )

            # tool_result capture stays HERE, after dispatch: this is the
            # first point the full result/error text exists. (The tool_call
            # record for this same call was already emitted above, BEFORE
            # dispatch -- see the comment at the top of this `for` body.)
            # Capture BEFORE _truncate_tool_result below: the log is the
            # lossless record, history is the capped view of it. This single
            # point covers every dispatch branch above -- builtin, MCP,
            # skill, runtime tools -- plus legacy review-hook refusals. A
            # continuation refusal was already finalized and recorded above.
            # Final-review IMPORTANT 3: tool_catalog.py documents `status` as
            # "ok or error", but this used to write only "ok" (any
            # "proceed" verdict, even a dispatch that actually failed -- see
            # `content = ... f"ERROR: {result.error}"` above) or "refused"
            # -- "error" was never reachable. `result` is only safe to read
            # here when verdict == "proceed": that is the sole branch above
            # that assigns it in THIS iteration (a non-"proceed" verdict
            # skips dispatch entirely, so `result` -- if it exists at all --
            # would be a stale value from a different call in this batch).
            if verdict == "proceed":
                record_status = "ok" if result.ok else "error"
            else:
                record_status = "refused"
            if continuation_checkpoint is not None:
                full_content = content
                continuation_cap = (
                    min(budget.max_tool_result_chars, 16_000)
                    if budget.max_tool_result_chars > 0
                    else 16_000
                )
                content = _truncate_tool_result(
                    content,
                    continuation_cap,
                    call.name,
                    total_limit=True,
                )
                target_state = (
                    "completed" if verdict == "proceed" and result.ok else "failed"
                )
                if not transition_call(
                    call,
                    target_state,
                    ContinuationResult(content),
                ):
                    return continuation_error()
                _emit_record(
                    deps,
                    "tool_result",
                    content=full_content,
                    tool=call.name,
                    status=record_status,
                    call_id=call.call_id,
                )
            else:
                record_number = _emit_record(
                    deps,
                    "tool_result",
                    content=content,
                    tool=call.name,
                    status=record_status,
                    call_id=call.call_id,
                )
                content = _truncate_tool_result(
                    content,
                    budget.max_tool_result_chars,
                    call.name,
                    record_number=record_number,
                )

            add(
                STEP_TOOL_RESULT,
                tool_name=call.name,
                result=content[:2000],
                tool_outcome=tool_outcome,
            )
            if restoring_batch and continuation_checkpoint is not None:
                if not expand_restore_history(continuation_checkpoint):
                    return continuation_error()
            else:
                _append_tool_result(messages, call, content)
