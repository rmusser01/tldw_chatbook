"""Pure data models for the agent runtime.

No Textual, app, DB, or I/O imports — see the vertical-slice spec
(Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

RUN_RUNNING = "running"
RUN_DONE = "done"
RUN_ERROR = "error"
RUN_STUCK = "stuck"
RUN_CANCELLED = "cancelled"
RUN_SUPERSEDED = "superseded"
TERMINAL_RUN_STATUSES = frozenset(
    {RUN_DONE, RUN_ERROR, RUN_STUCK, RUN_CANCELLED, RUN_SUPERSEDED})

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
RUNTIME_TOOL_NAMES = frozenset(
    {SPAWN_TOOL_NAME, FIND_TOOLS_NAME, LOAD_TOOLS_NAME})

DIRECT_DISCLOSE_THRESHOLD = 8
LOOP_DETECTION_N = 3


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


@dataclass(frozen=True)
class RunBudget:
    max_steps: int = 8
    max_wall_seconds: float = 240.0
    max_subagents: int = 2
    max_active_tools: int = 8
    max_subagent_result_chars: int = 4000


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


def clamp_child_budget(
    child: RunBudget, parent_remaining_seconds: float
) -> RunBudget:
    """Clamp a sub-agent's budget so it cannot outlive its parent.

    Wall-clock is clamped to the parent's remainder (floored at 1s);
    ``max_subagents`` is zeroed — depth-1 sub-agents never spawn.
    Steps are per-run and stay at the child's own default.
    """
    return RunBudget(
        max_steps=child.max_steps,
        max_wall_seconds=min(
            child.max_wall_seconds, max(parent_remaining_seconds, 1.0)),
        max_subagents=0,
        max_active_tools=child.max_active_tools,
        max_subagent_result_chars=child.max_subagent_result_chars,
    )
