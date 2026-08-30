"""Pure data models for the agent runtime.

No Textual, app, DB, or I/O imports — see the vertical-slice spec
(Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias

from tldw_chatbook.Chat.provider_continuation import (
    ContinuationResult,
    ProviderContinuationCheckpoint,
)

RUN_RUNNING = "running"
RUN_DONE = "done"
RUN_ERROR = "error"
RUN_STUCK = "stuck"
RUN_CANCELLED = "cancelled"
RUN_SUPERSEDED = "superseded"
TERMINAL_RUN_STATUSES = frozenset(
    {RUN_DONE, RUN_ERROR, RUN_STUCK, RUN_CANCELLED, RUN_SUPERSEDED}
)

AGENT_KIND_PRIMARY = "primary"
AGENT_KIND_SUBAGENT = "subagent"

STEP_MODEL = "model"
STEP_TOOL_CALL = "tool_call"
STEP_TOOL_RESULT = "tool_result"
STEP_SPAWN = "spawn"
STEP_ERROR = "error"
STEP_MODEL_REQUEST_STARTED = "model_request_started"
STEP_MODEL_RESPONSE_COMPLETED = "model_response_completed"
STEP_MODEL_RETRY = "model_retry"
STEP_MODEL_ERROR = "model_error"
STEP_MODEL_CANCELLED = "model_cancelled"
STEP_TOOL_PROPOSED = "tool_proposed"
STEP_APPROVAL_REQUESTED = "approval_requested"
STEP_APPROVAL_APPROVED = "approval_approved"
STEP_APPROVAL_DENIED = "approval_denied"
STEP_APPROVAL_REVOKED = "approval_revoked"
STEP_TOOL_EXECUTION_STARTED = "tool_execution_started"
STEP_TOOL_SUCCEEDED = "tool_succeeded"
STEP_TOOL_FAILED = "tool_failed"
STEP_TOOL_TIMED_OUT = "tool_timed_out"
STEP_TOOL_CANCELLED = "tool_cancelled"
# Fleet PR3b Task 1 (spec SS6): a steering entry delivered to a child at the
# protocol-coherent drain boundary records a step of this kind, so the step
# log shows WHEN each entry actually reached the model.
STEP_STEERING = "steering"

# Append-only agent-run lifecycle observations use dedicated storage-index
# bands. Control rows must remain below TRACE_STEP_INDEX_BASE; runtime trace
# rows and capture diagnostics use the following named bands.
TRACE_STEP_INDEX_BASE = 1_000_000
TRACE_CAPTURE_INDEX_BASE = 2_000_000
CONTROL_CAPTURE_INDEX_BASE = 3_000_000
# One control step can emit at most five trace observations (proposed,
# approval requested, approved/denied, execution started, terminal outcome),
# plus two one-time context observations per run. Keep the derived final trace
# index strictly below the capture band; owner_seq, not these indices, carries
# observation order.
MAX_RUN_CONTROL_STEPS = (
    TRACE_CAPTURE_INDEX_BASE - TRACE_STEP_INDEX_BASE - 3
) // 5
# Lifecycle stays above every runtime/capture band.
# keeping lifecycle at 10_000_000+ prevents collisions while owner_seq carries
# the real observation order independently of this storage identity.
AGENT_LIFECYCLE_INDEX_BASE = 10_000_000
STEP_AGENT_RUN_RESERVED = "agent_run_reserved"
STEP_AGENT_RUN_CREATED = "agent_run_created"
STEP_AGENT_RUN_RESUMED = "agent_run_resumed"
STEP_AGENT_RUN_STARTED = "agent_run_started"
STEP_AGENT_RUN_COMPLETED = "agent_run_completed"
STEP_AGENT_RUN_FAILED = "agent_run_failed"
STEP_AGENT_RUN_CANCELLED = "agent_run_cancelled"
STEP_AGENT_RUN_SUPERSEDED = "agent_run_superseded"

TOOL_OUTCOME_SUCCESS = "success"
TOOL_OUTCOME_FAILED = "failed"
TOOL_OUTCOME_BLOCKED = "blocked"
TOOL_OUTCOME_TIMEOUT = "timeout"
TOOL_OUTCOME_CANCELLED = "cancelled"
ToolOutcome: TypeAlias = Literal["success", "failed", "blocked", "timeout", "cancelled"]

# The two steering sources (spec SS6: "two paths, one mechanism"). The label
# the child sees is derived from the source by `format_steering_message`
# below -- prepended by the mechanism, never trusted from input.
STEERING_SOURCE_SUPERVISOR = "supervisor"
STEERING_SOURCE_USER = "user"
#: Cap on one steering entry's text, enforced by the producers at their own
#: boundaries (send_to_agent -- Task 2; the panel input -- Task 3). The
#: ``max_subagent_result_chars`` shape: a plain int ceiling, 4000.
MAX_STEERING_CHARS = 4000


def format_steering_message(source: str, text: str) -> str:
    """Render one steering entry exactly as the child's model will see it.

    THE one formatter (plan Task 1): the loop's history injection, the run
    log record, and every test render through this function, so the three
    can never drift. The label comes from ``source`` (one of
    ``STEERING_SOURCE_SUPERVISOR``/``STEERING_SOURCE_USER``); ``text`` is
    payload only -- a forged "[Steering from ...]" prefix inside it is
    still wrapped, never promoted to a label.

    Args:
        source: Who steered -- ``"supervisor"`` or ``"user"``.
        text: The steering message body.

    Returns:
        ``"[Steering from {source}] {text}"``.
    """
    return f"[Steering from {source}] {text}"


SPAWN_TOOL_NAME = "spawn_subagent"
FIND_TOOLS_NAME = "find_tools"
LOAD_TOOLS_NAME = "load_tools"
SKILL_FILE_TOOL_NAME = "skill_file"
INSTALL_SKILL_TOOL_NAME = "install_skill"
RUN_SKILL_SCRIPT_TOOL_NAME = "run_skill_script"
SEARCH_RUN_LOG_TOOL_NAME = "search_run_log"
# Phase 2 (run-log spec §10): the two aggregation/slicing runtime tools,
# registered exactly like SEARCH_RUN_LOG_TOOL_NAME above -- same name-
# constant + RUNTIME_TOOL_NAMES + tool_catalog schema + LoopDeps field +
# dispatch-branch + primary-agent-only-service-gate pattern. See
# agent_service.py's `log_active` gate and run_log_search.py's
# `compute_stats`/`slice_records` for the implementations these dispatch to.
RUN_LOG_STATS_TOOL_NAME = "run_log_stats"
RUN_LOG_SLICE_TOOL_NAME = "run_log_slice"
# Fleet (PR2a Task 6): the two tools a supervisor uses to manage children
# that now run CONCURRENTLY on their own threads instead of inline. Both
# are primary-only (pinned like install_skill) and both are dispatched
# IN-LOOP like spawn_subagent -- never through invoke_tool's per-call
# timeout wrapper, whose daemon thread would abandon a wait that is
# supposed to be bounded by the parent's own wall-clock instead.
WAIT_AGENTS_TOOL_NAME = "wait_agents"
CHECK_AGENTS_TOOL_NAME = "check_agents"
# Fleet steering (PR3b Task 2, spec SS6): the supervisor's producer for the
# per-child mailbox Task 1 added. Primary-only and fleet-gated like the two
# tools above, dispatched IN-LOOP like them (posting to a locked in-memory
# mailbox is instant, but the id-resolution copy lives in the service
# closure, not behind invoke_tool's daemon-thread timeout wrapper).
SEND_TO_AGENT_TOOL_NAME = "send_to_agent"
RUNTIME_TOOL_NAMES = frozenset(
    {
        SPAWN_TOOL_NAME,
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SKILL_FILE_TOOL_NAME,
        INSTALL_SKILL_TOOL_NAME,
        RUN_SKILL_SCRIPT_TOOL_NAME,
        SEARCH_RUN_LOG_TOOL_NAME,
        RUN_LOG_STATS_TOOL_NAME,
        RUN_LOG_SLICE_TOOL_NAME,
        WAIT_AGENTS_TOOL_NAME,
        CHECK_AGENTS_TOOL_NAME,
        SEND_TO_AGENT_TOOL_NAME,
    }
)

#: Above this, `initial_disclosure` defers everything to find_tools/
#: load_tools instead of direct-disclosing the whole catalog. Raised
#: alongside `RunBudget.max_active_tools` (8 -> 24) because the two are
#: coupled: `max_active_tools` is a one-way ratchet on the active set --
#: `load_tools` refuses a call that would exceed it with "no room", and
#: nothing ever unloads a tool once active -- so a catalog that clears the
#: raised ceiling but not this threshold would still pay for progressive
#: disclosure it can never actually use. The threshold has to rise with
#: the ceiling for the opposite reason too: `initial_disclosure` runs once
#: per RUN (every user message, not once per session), so a catalog sized
#: just above the OLD threshold paid a find_tools + load_tools round trip
#: before any real work, on every single message.
DIRECT_DISCLOSE_THRESHOLD = 16
LOOP_DETECTION_N = 3
#: Fence-protocol tool-result convention (`agent_runtime._append_tool_result`'s
#: fence branch: `{"role": "user", "content": f"{FENCE_TOOL_RESULT_PREFIX}
#: {call.name}: {content}"}`). Promoted to a shared constant (TASK-1272,
#: Phase 3) so `run_log_eviction`'s protocol-aware turn grouping can match
#: the exact same string rather than a re-typed copy that could silently
#: drift from the one `_append_tool_result` actually writes.
FENCE_TOOL_RESULT_PREFIX = "Tool result for "
#: Default ceiling on provider turns (STEP_MODEL steps) in one run. Stays
#: >= the default max_steps so it is provably unreachable at engine
#: defaults; it only becomes the operative limiter for a caller that raises
#: max_steps to match (see console_agent_bridge.CONSOLE_MAX_MODEL_TURNS).
DEFAULT_MAX_MODEL_TURNS = 30
# Longest tool-call cycle period the runtime detects (A->B->A->B is period 2).
MAX_LOOP_PERIOD = 4


@dataclass
class SkillFileBindings:
    """Per-run authorization + reader for the skill_file runtime tool.

    Mutable by design: seeded with the turn's $skill names; SkillRunner adds
    each spawned skill's name before spawn so a skill can always read its own
    bundle. Authorization lives here, never in config.allowed_tools.
    """

    authorized: set[str]
    reader: Callable[[str, str], dict] | None = None


@dataclass(frozen=True)
class ToolCatalogEntry:
    """One cheap-to-list catalog row: names and one-liners only."""

    id: str
    name: str
    one_line_description: str
    source: str


@dataclass(frozen=True)
class ToolSchema:
    """A tool's full definition, loaded on demand."""

    id: str
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    call_id: str = ""
    raw_arguments: str = ""


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str = ""
    error: str = ""
    # Optional refusal provenance lets the runtime distinguish a permission
    # block from an ordinary failed dispatch without interpreting payload text.
    outcome: ToolOutcome | None = None

    @classmethod
    def blocked(cls, error: str) -> ToolResult:
        """Return a permission/policy refusal with structured provenance.

        Args:
            error: User-visible refusal reason.

        Returns:
            A failed tool result explicitly classified as blocked.
        """
        return cls(ok=False, error=error, outcome=TOOL_OUTCOME_BLOCKED)


@dataclass(frozen=True)
class ContinuationEventContext:
    """Durable owner and run identity for one continuation lineage."""

    owner_message_id: str | None
    run_id: str
    agent_kind: Literal["primary", "subagent", "fleet"]
    durability: Literal["persistent", "ephemeral"]


@dataclass(frozen=True)
class ToolBatchReady:
    """A complete canonical call batch is ready for durable creation/update."""

    context: ContinuationEventContext
    checkpoint: ProviderContinuationCheckpoint
    expected_checkpoint_revision: int | None


@dataclass(frozen=True)
class ToolCallExecuting:
    """One call is about to cross its dispatch boundary."""

    context: ContinuationEventContext
    call_id: str
    expected_checkpoint_revision: int


@dataclass(frozen=True)
class ToolCallFinished:
    """One exact provider-bound result is ready for durable storage."""

    context: ContinuationEventContext
    call_id: str
    expected_checkpoint_revision: int
    target_state: Literal["completed", "failed"]
    result: ContinuationResult


@dataclass(frozen=True)
class FinalContinuation:
    """A tool-free final answer and complete checkpoint are ready together."""

    context: ContinuationEventContext
    checkpoint: ProviderContinuationCheckpoint
    expected_checkpoint_revision: int | None
    assistant_content: str


ProviderContinuationEvent: TypeAlias = (
    ToolBatchReady | ToolCallExecuting | ToolCallFinished | FinalContinuation
)


@dataclass(frozen=True)
class ModelTurn:
    """One provider response: raw text plus any native tool calls.

    ``assistant_message`` carries the provider-shaped assistant message for
    native tool-call turns (content plus the raw ``tool_calls`` array,
    echoed verbatim into history so the follow-up ``role="tool"`` results
    pair with their calls by id). ``None`` for fence-protocol turns, whose
    history keeps the plain-text convention.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    assistant_message: dict | None = None
    tokens: int = 0
    provider_continuation: ProviderContinuationCheckpoint | None = None


@dataclass(frozen=True)
class RunBudget:
    """Caps bounding one agent run: steps, wall-clock, sub-agents, and
    (task-244) model-turn count.

    ``max_model_turns`` counts ``STEP_MODEL`` steps and is checked in
    ``agent_runtime.run_agent_loop`` immediately after the step-budget
    check. At the defaults below it EXCEEDS ``max_steps``, which makes it
    provably unreachable *at engine defaults*: every model turn appends at
    least one step (the ``STEP_MODEL`` step itself), so with
    ``max_model_turns >= max_steps`` the step-budget check always fires
    first — engine-default behavior stays byte-identical to the
    pre-task-244 loop. The cap only becomes the operative limiter for a
    caller that also raises ``max_steps`` (the Console bridge does; see
    ``console_agent_bridge.CONSOLE_RUN_BUDGET``).
    """

    max_steps: int = 8
    max_wall_seconds: float = 240.0
    max_subagents: int = 2
    # Raised 8 -> 24 alongside DIRECT_DISCLOSE_THRESHOLD (8 -> 16); see that
    # constant's comment for why the two move together. This ceiling is
    # itself a one-way ratchet within a run: load_tools() refuses a call
    # that would exceed it ("no room") and nothing ever unloads an active
    # tool, so raising it only ever widens what a run can reach, never
    # narrows it back down mid-run.
    max_active_tools: int = 24
    max_subagent_result_chars: int = 4000
    # Ceiling on how much of ONE tool result enters conversation history.
    # Enforced at the history-append seam (agent_runtime), NOT per tool, so
    # built-in, MCP, and skill results are all bounded by the same rule.
    # Derived from max_subagent_result_chars (4000): four times a whole
    # sub-agent result is generous for a single call while keeping a
    # 30-turn run tractable. 0 = unlimited, restoring pre-cap behaviour.
    max_tool_result_chars: int = 16000
    # Primary provider-call limiter (task-244): counts STEP_MODEL turns.
    # Raised 8 -> 20, then 20 -> 30, so an agent gets ~30 tool-calling
    # rounds per user message. It stays >= max_steps at engine defaults, so
    # it remains provably unreachable here (every model turn appends >=1
    # step, so the step check fires first) — engine-default behavior is
    # unchanged. It bites only where max_steps is raised to match; the
    # Console bridge sizes both together (CONSOLE_RUN_BUDGET).
    max_model_turns: int = DEFAULT_MAX_MODEL_TURNS
    # task-326: cumulative prompt+completion token spend ceiling for one run.
    # 0 = unlimited (default), keeping existing runs byte-identical. This is a
    # SPEND ceiling (the growing prompt is re-sent each call), not a window size.
    max_total_tokens: int = 0
    # task-327: per-tool-call EXECUTION-time ceiling. A single
    # custom/blocking tool provider must not be able to wedge a
    # cooperative-cancel run forever. 0 = unlimited (opt-out). Enforced in
    # agent_service's impure seam (the pure runtime stays timeout-free).
    # MCP tools DO flow through this same wrapper (MCPToolProvider is
    # registered into the same per-run ToolCatalogRegistry as builtins),
    # so the default sits above MCP's own execution worst case: up to 65s
    # to execute (`_tool_call_timeout() = 60.0` in
    # `MCP/unified_control_plane_service.py` plus
    # `_RESULT_WAIT_SLACK_SECONDS = 5.0` in `Agents/mcp_tool_provider.py`).
    # ADR-067: time spent waiting on a HUMAN decision no longer counts --
    # the wrapper's deadline pauses while `Agents.human_input_wait` marks
    # the run (approval card / skill confirms; their default is no
    # auto-deny at all), superseding the old requirement that the approval
    # timeout stay strictly below this ceiling. `[mcp]
    # tool_call_timeout_seconds` is still user-tunable: raising it past
    # this default can still reopen the double-execution window (the
    # wrapper reporting "timed out" for a call that later really executes
    # on its abandoned thread -- see `_call_with_timeout`'s docstring).
    max_tool_call_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_steps > MAX_RUN_CONTROL_STEPS:
            raise ValueError(
                f"max_steps must be <= {MAX_RUN_CONTROL_STEPS} to preserve "
                "agent trace storage bands"
            )


#: Fleet spec §4: validation caps for user-authored agent definitions.
#: description rides the spawn tool's schema (re-sent every fence-model
#: turn); instructions ride every child model turn — both caps are cost
#: controls, not polish.
AGENT_DEFINITION_NAME_PATTERN = r"^[a-z][a-z0-9-]{0,63}$"
AGENT_DEFINITION_RESERVED_NAMES = frozenset({"general", "subagent"})
AGENT_DEFINITION_DESCRIPTION_MAX_CHARS = 200
AGENT_DEFINITION_INSTRUCTIONS_MAX_CHARS = 16_000


@dataclass(frozen=True)
class AgentDefinition:
    """A named, user-authored sub-agent template (fleet spec §4).

    ``instructions`` are APPENDED to the internal ``agents.subagent_system``
    prompt at spawn time — never a replacement (the base prompt is an
    identity contract: console_agent_bridge detects sub-agent turns by
    prefix-matching it). ``tool_allowlist`` only ever narrows the child's
    inherited allow-list (intersection, never union); empty means inherit.
    ``model`` overrides the parent's model on the SAME provider endpoint;
    empty means inherit.
    """

    name: str
    description: str = ""
    instructions: str = ""
    tool_allowlist: tuple[str, ...] = ()
    model: str = ""
    enabled: bool = True


def validate_agent_definition(defn: AgentDefinition) -> list[str]:
    """Return validation errors for ``defn``; empty list means valid."""
    errors: list[str] = []
    if not re.fullmatch(AGENT_DEFINITION_NAME_PATTERN, defn.name or ""):
        errors.append(
            "name must be a lowercase slug (a-z, 0-9, hyphens; starts with "
            "a letter; max 64 chars)"
        )
    if defn.name in AGENT_DEFINITION_RESERVED_NAMES:
        errors.append(f"name '{defn.name}' is reserved")
    if len(defn.description) > AGENT_DEFINITION_DESCRIPTION_MAX_CHARS:
        errors.append(
            f"description exceeds {AGENT_DEFINITION_DESCRIPTION_MAX_CHARS} chars"
        )
    if "\n" in defn.description:
        # build_spawn_schema renders description into a "- name — desc"
        # roster line (Agents/tool_catalog.py); an embedded newline could
        # forge extra roster lines the supervisor reads as real entries.
        errors.append("description must be a single line")
    if not defn.instructions.strip():
        errors.append("instructions must not be empty")
    if len(defn.instructions) > AGENT_DEFINITION_INSTRUCTIONS_MAX_CHARS:
        errors.append(
            f"instructions exceed {AGENT_DEFINITION_INSTRUCTIONS_MAX_CHARS} chars"
        )
    return errors


def definition_fingerprint(defn: AgentDefinition) -> str:
    """16-hex-char content hash of the fields that shape a child run.

    Covers instructions/tool_allowlist/model ONLY — the audit identity of
    what actually ran (spec §4). description/enabled are presentation.
    """
    payload = json.dumps(
        {
            "instructions": defn.instructions,
            "tool_allowlist": sorted(defn.tool_allowlist),
            "model": defn.model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def definition_from_row(row: dict) -> AgentDefinition:
    """Build an ``AgentDefinition`` from an ``agent_definitions`` DB row
    (``tool_allowlist`` already JSON-decoded to a list by the DB layer)."""
    return AgentDefinition(
        name=row["name"],
        description=row["description"],
        instructions=row["instructions"],
        tool_allowlist=tuple(row["tool_allowlist"]),
        model=row["model"],
        enabled=bool(row["enabled"]),
    )


@dataclass
class AgentStep:
    index: int
    kind: str
    summary: str = ""
    tool_name: str = ""
    args: dict | None = None
    result: str = ""
    created_at: str = ""
    # Optional for backward compatibility with persisted steps written before
    # tool outcomes were structured. Only meaningful on STEP_TOOL_RESULT.
    tool_outcome: ToolOutcome | None = None
    status: str = ""
    parent_event_id: str | None = None
    source_event_id: str | None = None
    replacement_event_id: str | None = None
    field_states: dict[str, str] = field(default_factory=dict)
    sensitivity: str = ""
    # Trace-v2 envelope fields. ``index`` remains the legacy control-step
    # identity; owner_seq is the observation order across control + lifecycle.
    owner_seq: int | None = None
    call_id: str = ""
    parent_step_index: int | None = None
    source_step_index: int | None = None


@dataclass(frozen=True)
class AgentConfig:
    """One run's model, prompt, tool allow-list and budget.

    Attributes:
        model: The model id handed to ``chat_api_call``.
        system_prompt: The run's system prompt, before any protocol or
            run-log section the service appends.
        allowed_tools: **The CATALOG allow-list — it does not govern the
            runtime tools.** TASK-16788 recorded this deliberately rather
            than "fixing" it; the contract is:

            *What it governs.* Every name reached through the tool
            CATALOG (``ToolCatalogRegistry``: builtins, local tools,
            Library tools, MCP tools, skill tools). ``AgentService.
            _run_one`` filters the initial disclosure by it (Q7(a)), the
            ``find_tools`` and ``load_schemas`` closures re-filter by it
            (Q7(b)/(c)), and ``_make_invoke_tool``'s ``invoke_tool``
            refuses a call whose name is outside it. An empty tuple
            therefore means "no catalog tool at all", disclosed or
            callable.

            *What it does NOT govern.* The runtime tools --
            ``RUNTIME_TOOL_NAMES`` above: ``spawn_subagent``,
            ``wait_agents``/``check_agents``/``send_to_agent``,
            ``find_tools``/
            ``load_tools``, ``skill_file``, ``install_skill``,
            ``run_skill_script``, and ``search_run_log``/
            ``run_log_stats``/``run_log_slice``. These are not catalog
            entries; ``_run_one`` appends their schemas to
            ``runtime_schemas`` AFTER the allow-list filter, each under
            its OWN gate (``max_subagents > 0``; a live fleet; the
            progressive-disclosure ``offer_find_load``; authorized skill
            bindings; the primary-only ``install_skill`` wiring; the
            run-script wiring; ``log_active`` = primary + an active run-log
            writer + something else already disclosed). So a run with an
            EMPTY allow-list is still offered whichever runtime tools its
            own gates admit -- pinned by ``Tests/Agents/
            test_run_log_service_wiring.py::
            test_run_log_tools_are_offered_under_an_empty_allow_list``.

            *Why calls are not caught later.* ``run_agent_loop`` dispatches
            each runtime name in its own dedicated ``elif`` branch BEFORE
            the generic ``deps.invoke_tool`` fallback, so ``invoke_tool``'s
            allow-list check never sees a runtime call the loop is WIRED for
            (8 of the 11 branches are guarded by their dep being
            non-None: an unwired runtime name -- e.g. a sub-agent's
            search_run_log -- falls through to the catalog fallback
            and is refused there). The
            one exception is ``spawn_subagent``, whose branch re-checks
            ``config.allowed_tools`` itself and refuses before dispatch
            (Q6) -- the rest are governed only by their gates and by the
            permission layer.

            *Consequence for callers.* An embedder that narrows
            ``allowed_tools`` to isolate an experiment's tool set does NOT
            get an exhaustive restriction: the run-log tools in particular
            can still consume agent steps. That confound is recorded in
            ``Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/
            report.md`` (TASK-16174's oracle run, whose tool-OFF arm ended
            question q3 ``stuck`` on run-log calls). To hold a runtime tool
            out of a run, close its own gate (e.g. ``max_subagents=0``,
            or leave the run-log writer inactive) rather than the
            allow-list.
        budget: This run's caps (see ``RunBudget``).
        native_tools: Whether to use provider-native tool-calling when the
            endpoint supports it; ``False`` forces the fence protocol.
        workspace_context_note: Optional environment note appended to the
            system prompt each turn when the run is bound to a NON-default
            workspace (built by ``workspace_file_roots.workspace_context_note``
            with launch-relative, never absolute, paths). Empty for the default
            workspace, so the common case adds nothing. Carried on the config
            so it propagates verbatim onto spawned sub-agents' configs.
        personal_context_block: Immutable, already-authorized user-owned data
            block appended to every model request in this run tree. Empty by
            default so existing request bytes are unchanged.
        response_reserve_tokens: Non-negative output-token capacity excluded
            from project-instruction input admission.
    """

    model: str
    system_prompt: str
    allowed_tools: tuple[str, ...] = ()
    budget: RunBudget = field(default_factory=RunBudget)
    native_tools: bool = True
    workspace_context_note: str = ""
    personal_context_block: str = ""
    response_reserve_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.response_reserve_tokens < 0:
            raise ValueError("response_reserve_tokens must be non-negative")


@dataclass
class RunOutcome:
    status: str
    steps: list[AgentStep]
    final_text: str = ""
    subagents_spawned: int = 0
    total_tokens: int = 0
    # PR3b Task 4 (finished-agent continuation, spec SS6): the run's
    # message history sliced at the last protocol-coherent drain boundary
    # (``run_agent_loop``'s ``coherent_len``), plus -- on RUN_DONE only --
    # the final assistant text the done-return excludes. By construction
    # this can never end inside a split native ``tool_calls`` <->
    # ``role:"tool"`` batch (the mid-batch cancel/cycle-stuck returns
    # slice the whole in-flight batch away), so it is safe to seed a
    # resumed child's first provider call with. IN-MEMORY ONLY: the
    # coordinator's retention store reads it off the outcome;
    # ``AgentService._persist`` never writes it to the database.
    final_messages: list[dict] | None = None


def clamp_child_budget(child: RunBudget, parent_remaining_seconds: float) -> RunBudget:
    """Clamp a TURN-SCOPED/INLINE sub-agent's budget so it cannot outlive its parent.

    Scope, stated explicitly (PR3a-1 Task 5 review, Defect 1): this is
    NOT a system-wide "a child can never outlive its parent" invariant --
    that claim is false of this system since PR3a-1 Task 2 let a threaded
    child survive past `_settle_fleet`. This function's real, current
    scope is production's call site for exactly ONE spawn path: a
    turn-scoped or explicitly ``inline=True`` child, which blocks the
    parent inside ``AgentService.spawn`` and has no `_settle_fleet` to
    bound it externally (``spawn``'s own branch on
    ``fleet is None or inline`` -- the identical predicate its dispatch
    below tests). A THREADED, non-inline child -- the one kind that can
    actually survive past `_settle_fleet` -- never reaches this function;
    it goes through ``contain_child_budget`` below instead, which has its
    own independent ceiling and makes no parent-outlive-proof claim at
    all.

    (An earlier draft of this docstring said this function was "no
    longer production's call site" -- that was true only briefly, for a
    version of Task 5 that applied `contain_child_budget` unconditionally
    and broke the turn-scoped/inline path's own byte-identical-behaviour
    guarantee. Corrected: this function IS still production's call site,
    for the turn-scoped/inline half of `spawn`'s branch.)

    Sub-agents deliberately INHERIT ``max_model_turns`` and ``max_steps``
    rather than being clamped down (operator decision, 2026-07-25, first
    taken when the Console cap was raised 8 -> 20 and re-confirmed when it
    went 20 -> 30). A child therefore gets the same round budget as its
    parent, so one Console message can reach
    ``max_model_turns * (1 + max_subagents)`` provider turns in the worst
    case — 90 at the Console's current 30/2. For a child THIS FUNCTION
    bounds (turn-scoped/inline only), that worst case is bounded in TIME
    by the wall-clock clamp below (such a child can never outlive its
    parent's remaining budget) -- but that TIME bound does NOT hold for a
    threaded survivor candidate, which is bounded by
    ``contain_child_budget``'s own independent ceiling instead (see that
    function's docstring for the resulting worst-case aggregate).
    ``max_total_tokens`` is passed through UNCHANGED below, not divided
    among children, so it bounds each run's OWN spend independently, not
    the aggregate: the parent and each of up to ``max_subagents`` children
    can each spend up to that ceiling, for a real worst-case aggregate of
    roughly ``(1 + max_subagents)x`` it — not a value it bounds directly
    (the Console sets it non-zero for exactly this containment reason
    regardless — see ``console_agent_bridge.CONSOLE_MAX_TOTAL_TOKENS`` for
    its concrete numbers). Do not "fix" this by clamping turns without
    checking the inherit-turns decision above.

    Wall-clock is clamped to the parent's remainder (floored at 1s);
    ``max_subagents`` is zeroed — depth-1 sub-agents never spawn.
    Steps are per-run and stay at the child's own default.
    """
    return RunBudget(
        max_steps=child.max_steps,
        max_wall_seconds=min(
            child.max_wall_seconds, max(parent_remaining_seconds, 1.0)
        ),
        max_subagents=0,
        max_active_tools=child.max_active_tools,
        max_subagent_result_chars=child.max_subagent_result_chars,
        max_tool_result_chars=child.max_tool_result_chars,
        max_model_turns=child.max_model_turns,
        max_total_tokens=child.max_total_tokens,
        max_tool_call_seconds=child.max_tool_call_seconds,
    )


def contain_child_budget(child: RunBudget, max_wall_seconds: float) -> RunBudget:
    """Bound a THREADED SURVIVOR CANDIDATE's own budget -- independent of its parent.

    Scope, stated explicitly (PR3a-1 Task 5 review, Defect 1): this does
    NOT replace ``clamp_child_budget``'s "child can never outlive its
    parent" clamp everywhere -- ``AgentService.spawn`` BRANCHES on
    ``fleet is None or inline``, and a turn-scoped or explicitly
    ``inline=True`` child still goes through ``clamp_child_budget``,
    unmodified, exactly as before this task (that child blocks the
    parent and has no `_settle_fleet` to bound it externally, so it must
    keep byte-identical turn-scoped behaviour -- an earlier draft of this
    task applied THIS function unconditionally and broke that; see
    ``clamp_child_budget``'s own docstring for the correction). This
    function is production's call site for the OTHER half of that
    branch only: a THREADED, non-inline child -- the one kind that can
    actually survive past `_settle_fleet` (PR3a-1 Task 2, spec Sec 5
    "Containment") -- for which a still-``running`` child is expected
    background work, not a dead attempt. Tying THAT child's wall-clock
    ceiling to how much of the PARENT's own budget happened to be left
    at spawn time made its effective bound an accident of WHEN in the
    turn it was spawned -- a child spawned in the run's last second would
    have gotten almost no time of its own, and a child spawned early
    would have inherited most of the parent's, neither of which
    describes a bound on the CHILD's own work.

    ``run_agent_loop``'s own wall-clock check (``agent_runtime.py``) is
    already measured from the RUN'S OWN ``started``, not the parent's, so
    handing a child a plain, caller-resolved ceiling here needs no
    engine-side change -- only the caller (``AgentService.spawn``, for
    its threaded/non-inline branch only) stops deriving it from the
    parent's remainder and instead resolves it from
    ``[agents] child_max_wall_seconds`` (default
    ``agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS``; see that constant's
    own comment for the sizing rationale and the resulting worst-case
    aggregate).

    Containment is bounded in three dimensions, none of them the
    parent's lifetime: TIME (this function's own ``max_wall_seconds``
    argument -- genuinely independent of the parent, this task's actual
    fix; caveat easy to miss: a child blocked INSIDE one provider call is
    not stopped by its own wall clock at all -- ``run_agent_loop``'s
    check only runs BETWEEN loop iterations, before each
    ``deps.call_model``, so a hung provider call can hold a child open
    past this ceiling until that call itself returns), SPEND
    (``max_total_tokens``, passed through UNCHANGED below
    exactly as ``clamp_child_budget`` already did -- see that function's
    own docstring for why the real worst-case aggregate spend is
    ``(1 + max_subagents)x`` that ceiling, not a value it bounds
    directly; this task does not change that math), and COUNT --
    ``[agents] max_live_subagents``, enforced by ``FleetCoordinator.
    reserve``. Read its scope precisely, because it changed twice and
    the first version of this docstring got it wrong: it bounds live
    children per COORDINATOR, and a coordinator's lifetime belongs to
    whoever owns it. For a bare ``AgentService`` with no injected
    coordinator that is still ONE ``run_turn`` call -- which was ALSO
    Console's situation until PR3a-1 Task 6a, and is why two consecutive
    turns each spawning 2 blocking children ran 4 at once against a cap
    of 2 (Task 5 review, Defect 2, disproved by execution, not
    argument). Task 6a gave Console a coordinator per CONVERSATION,
    owned by ``ConsoleAgentBridge`` and injected into every service it
    builds, so there the cap now holds across turns; nothing caps the
    aggregate across conversations or across processes. This function
    does not participate in any of that -- it is stated here only so the
    third dimension is not read as bounded by something it is not.

    Sub-agents still deliberately INHERIT ``max_model_turns`` and
    ``max_steps`` unchanged (the same 2026-07-25 operator decision
    ``clamp_child_budget`` documents) -- only ``max_wall_seconds`` and
    ``max_subagents`` are ever touched here.

    ``clamp_child_budget`` above is UNCHANGED and remains a live
    production call site (for the OTHER, turn-scoped/inline branch of
    ``AgentService.spawn`` -- see its own docstring). Keeping the two
    functions separate, rather than branching inside one, avoids forcing
    a single function to serve both the parent-remainder-clamp shape and
    this independent-ceiling shape through a conditional -- and it is
    ``AgentService.spawn`` itself that branches between them, so each
    function's contract stays a flat, unconditional description of ONE
    path.

    Args:
        child: The would-be child's budget (today: the parent's own
            ``config.budget``, since sub-agents inherit steps/turns/tokens
            from the parent -- see ``AgentService.spawn``).
        max_wall_seconds: The child's own wall-clock ceiling, resolved by
            the caller -- counted from the CHILD's own start, never from
            the parent's, and never shrunk by how much of the parent's own
            budget remains.

    Returns:
        A new ``RunBudget`` with ``max_wall_seconds`` set to the given
        ceiling (floored at 1s, same floor ``clamp_child_budget`` uses)
        and ``max_subagents`` zeroed -- depth-1 sub-agents never spawn,
        an invariant this function preserves exactly, not just by
        omission. Every other field passes through unchanged.

        NaN and infinite input are both treated as invalid and floored to
        1s exactly like any other non-positive value, not passed through:
        ``max(float("nan"), 1.0)`` evaluates to ``nan`` in Python (``1.0 >
        nan`` is ``False``, so ``max`` keeps its first argument), and
        ``deps.clock() - started > nan`` -- ``run_agent_loop``'s own
        wall-clock check -- is then always ``False``, silently disabling
        the ceiling entirely rather than flooring it. A plain
        ``max(..., 1.0)`` alone does not defend against that; it only
        looks like it does. This function's only production call site
        (the threaded/non-inline branch of ``AgentService.spawn``)
        already can't reach this, since ``_coerce_child_max_wall_seconds``
        falls back to the config default before a non-finite value ever
        gets here -- but this function's own floor must hold for ANY
        caller regardless of argument provenance, not rely on a single
        upstream guard staying in place forever.
    """
    if not math.isfinite(max_wall_seconds):
        max_wall_seconds = 1.0
    return RunBudget(
        max_steps=child.max_steps,
        max_wall_seconds=max(max_wall_seconds, 1.0),
        max_subagents=0,
        max_active_tools=child.max_active_tools,
        max_subagent_result_chars=child.max_subagent_result_chars,
        max_tool_result_chars=child.max_tool_result_chars,
        max_model_turns=child.max_model_turns,
        max_total_tokens=child.max_total_tokens,
        max_tool_call_seconds=child.max_tool_call_seconds,
    )
