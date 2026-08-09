"""Pure data models for the agent runtime.

No Textual, app, DB, or I/O imports — see the vertical-slice spec
(Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable

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


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str = ""
    error: str = ""


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
    # task-327: per-tool-call wall-clock ceiling. A single custom/blocking
    # tool provider must not be able to wedge a cooperative-cancel run
    # forever. 0 = unlimited (opt-out). Enforced in agent_service's impure
    # seam (the pure runtime stays timeout-free). MCP tools DO flow through
    # this same wrapper (MCPToolProvider is registered into the same
    # per-run ToolCatalogRegistry as builtins), so the default is set
    # deliberately above MCP's own worst case rather than independent of
    # it: an "ask"-gated call can wait up to ~121s for human approval
    # (`_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 120.0` in
    # `Chat/console_chat_controller.py`, polled every
    # `_MCP_APPROVAL_POLL_SECONDS = 1.0`) and then up to 65s to execute
    # (`_tool_call_timeout() = 60.0` in
    # `MCP/unified_control_plane_service.py` plus
    # `_RESULT_WAIT_SLACK_SECONDS = 5.0` in `Agents/mcp_tool_provider.py`)
    # -- ~186s end to end AT MCP'S DEFAULT CONFIG. Both cited MCP bounds are
    # user-tunable (`[mcp] tool_call_timeout_seconds`, and the approval
    # timeout resolved from config in `console_chat_controller`), so this
    # invariant only holds while MCP itself stays at its shipped defaults --
    # a user who raises either MCP-side bound can still reopen the
    # double-execution window this default is meant to avoid. Lowering this
    # below that risks the wrapper reporting "timed out" for a call that
    # later really executes on its abandoned thread (see
    # `_call_with_timeout`'s docstring).
    max_tool_call_seconds: float = 300.0


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


@dataclass(frozen=True)
class AgentConfig:
    model: str
    system_prompt: str
    allowed_tools: tuple[str, ...] = ()
    budget: RunBudget = field(default_factory=RunBudget)
    native_tools: bool = True


@dataclass
class RunOutcome:
    status: str
    steps: list[AgentStep]
    final_text: str = ""
    subagents_spawned: int = 0
    total_tokens: int = 0


def clamp_child_budget(child: RunBudget, parent_remaining_seconds: float) -> RunBudget:
    """Clamp a sub-agent's budget so it cannot outlive its parent.

    Sub-agents deliberately INHERIT ``max_model_turns`` and ``max_steps``
    rather than being clamped down (operator decision, 2026-07-25, first
    taken when the Console cap was raised 8 -> 20 and re-confirmed when it
    went 20 -> 30). A child therefore gets the same round budget as its
    parent, so one Console message can reach
    ``max_model_turns * (1 + max_subagents)`` provider turns in the worst
    case — 90 at the Console's current 30/2. That worst case is bounded in
    TIME by the wall-clock clamp below (a child can never outlive its
    parent's remaining budget). ``max_total_tokens`` is passed through
    UNCHANGED below, not divided among children, so it bounds each run's
    OWN spend independently, not the aggregate: the parent and each of up
    to ``max_subagents`` children can each spend up to that ceiling, for a
    real worst-case aggregate of roughly ``(1 + max_subagents)x`` it — not
    a value it bounds directly (the Console sets it non-zero for exactly
    this containment reason regardless — see
    ``console_agent_bridge.CONSOLE_MAX_TOTAL_TOKENS`` for its concrete
    numbers). Do not "fix" this by clamping turns without checking the
    inherit-turns decision above.

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
