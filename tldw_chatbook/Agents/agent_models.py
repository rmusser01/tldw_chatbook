"""Pure data models for the agent runtime.

No Textual, app, DB, or I/O imports — see the vertical-slice spec
(Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md).
"""

from __future__ import annotations

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
RUNTIME_TOOL_NAMES = frozenset(
    {
        SPAWN_TOOL_NAME,
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SKILL_FILE_TOOL_NAME,
        INSTALL_SKILL_TOOL_NAME,
        RUN_SKILL_SCRIPT_TOOL_NAME,
    }
)

DIRECT_DISCLOSE_THRESHOLD = 16
LOOP_DETECTION_N = 3
#: Default ceiling on provider turns (STEP_MODEL steps) in one run. Stays
#: >= the default max_steps so it is provably unreachable at engine
#: defaults; it only becomes the operative limiter for a caller that raises
#: max_steps to match (see console_agent_bridge.CONSOLE_MAX_MODEL_TURNS).
DEFAULT_MAX_MODEL_TURNS = 20
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
    # Raised 8 -> 20 so an agent gets ~20 tool-calling rounds per user
    # message rather than ~8. It stays >= max_steps at engine defaults, so
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
    rather than being clamped down (operator decision, 2026-07-25, when the
    Console cap was raised 8 -> 20). A child therefore gets the same
    round budget as its parent, so one Console message can reach
    ``max_model_turns * (1 + max_subagents)`` provider turns in the worst
    case — 60 at the Console's current 20/2. That worst case is bounded in
    TIME by the wall-clock clamp below (a child can never outlive its
    parent's remaining budget), not in spend. Do not "fix" this by
    clamping turns without checking that decision.

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
        max_model_turns=child.max_model_turns,
        max_total_tokens=child.max_total_tokens,
        max_tool_call_seconds=child.max_tool_call_seconds,
    )
