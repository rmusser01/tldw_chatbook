# Agent Runtime — Plan A: Engine + Persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the agent runtime's engine and persistence — pure models, fence-first tool protocol, the constrained control loop, the ToolProvider catalog with progressive disclosure, `AgentRunsDB`, and the service that wires them to `chat_api_call` — with **no Console UI** (that is Plan B).

**Architecture:** A pure control loop (`agent_runtime.py`) driven entirely by injected callables, fed by a `ToolProvider` registry (`tool_catalog.py`) and persisted to a dedicated small DB (`AgentRuns_DB.py`); `agent_service.py` is the only impure module, adapting `chat_api_call` + `tool_executor` + the DB to the loop's callables. Spec: `Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md` — read it once before Task 1; it is the requirements document.

**Tech Stack:** Python ≥3.11, dataclasses, sqlite3 via the repo's `BaseDB` pattern, pytest.

## Global Constraints

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime` (branch `claude/agent-runtime-spec`). All paths below are relative to it.
- Tests ONLY via the venv python: `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` → `$PY -m pytest …`. No `AppTest` (doesn't exist in this Textual version); no UI tests in Plan A at all.
- **Pure modules** (`agent_models.py`, `agent_runtime.py`) must have NO imports of Textual, the app, any DB, or any I/O library. `tool_catalog.py` may import `tldw_chatbook.Tools.tool_executor` (its wrapping job) but nothing UI/DB. Only `agent_service.py` touches `chat_api_call` and the DB.
- Fence protocol: the fence is ` ```tool_call ` and MUST be the first non-whitespace content of a response to count as a leading tool call. Malformed/partial JSON parses to `None` — never an exception.
- Loop detection: `N = 3` identical consecutive calls, identical = same `(tool_name, json.dumps(args, sort_keys=True))`.
- Small-catalog direct-disclose threshold: `8`. When direct-disclosed, `find_tools`/`load_tools` are NOT offered.
- Budgets: `RunBudget(max_steps=8, max_wall_seconds=240.0, max_subagents=2, max_active_tools=8, max_subagent_result_chars=4000)` defaults. Child budget: wall-clock clamped to the parent's remainder; `max_subagents=0` (depth 1).
- `RunStatus` values exactly: `running|done|error|stuck|cancelled|superseded`.
- SQL always parameterized; DB writes via the `transaction()` context manager; the `BaseDB`/`WorkspaceDB` pattern opens a fresh connection per call (this satisfies the spec's separate-read-connection rule).
- **Provider fact (verified):** `Chat_Functions.py`'s llama_cpp param map has `'tools'` commented out — the QA provider does NOT pass native tools. The service therefore ALWAYS uses the text protocol; the loop's native `ModelTurn.tool_calls` path exists and is unit-tested, but no service wiring feeds it in Plan A.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Pure models (`agent_models.py`)

**Files:**
- Create: `tldw_chatbook/Agents/__init__.py` (empty)
- Create: `tldw_chatbook/Agents/agent_models.py`
- Test: `Tests/Agents/__init__.py` (empty), `Tests/Agents/test_agent_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by every later task, exact names): constants `RUN_RUNNING/RUN_DONE/RUN_ERROR/RUN_STUCK/RUN_CANCELLED/RUN_SUPERSEDED`, `TERMINAL_RUN_STATUSES`, `AGENT_KIND_PRIMARY/AGENT_KIND_SUBAGENT`, `STEP_MODEL/STEP_TOOL_CALL/STEP_TOOL_RESULT/STEP_SPAWN/STEP_ERROR`, `SPAWN_TOOL_NAME="spawn_subagent"`, `FIND_TOOLS_NAME="find_tools"`, `LOAD_TOOLS_NAME="load_tools"`, `RUNTIME_TOOL_NAMES`, `DIRECT_DISCLOSE_THRESHOLD=8`, `LOOP_DETECTION_N=3`; dataclasses `ToolCatalogEntry(id,name,one_line_description,source)`, `ToolSchema(id,name,description,parameters)`, `ToolCall(name,args,call_id="")`, `ToolResult(ok,content="",error="")`, `ModelTurn(text="",tool_calls=())`, `RunBudget(...)`, `AgentStep(index,kind,summary="",tool_name="",args=None,result="",created_at="")`, `AgentConfig(model,system_prompt,allowed_tools=(),budget=RunBudget())`, `RunOutcome(status,steps,final_text="",subagents_spawned=0)`; function `clamp_child_budget(child, parent_remaining_seconds) -> RunBudget`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Agents/test_agent_models.py
"""Pure model tests: values, defaults, and the child-budget clamp."""
import dataclasses

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT, DIRECT_DISCLOSE_THRESHOLD,
    FIND_TOOLS_NAME, LOAD_TOOLS_NAME, LOOP_DETECTION_N, RUN_CANCELLED,
    RUN_DONE, RUN_ERROR, RUN_RUNNING, RUN_STUCK, RUN_SUPERSEDED,
    RUNTIME_TOOL_NAMES, SPAWN_TOOL_NAME, TERMINAL_RUN_STATUSES, AgentConfig,
    AgentStep, ModelTurn, RunBudget, RunOutcome, ToolCall, ToolCatalogEntry,
    ToolResult, ToolSchema, clamp_child_budget,
)


def test_run_status_values_and_terminal_set():
    assert (RUN_RUNNING, RUN_DONE, RUN_ERROR, RUN_STUCK, RUN_CANCELLED,
            RUN_SUPERSEDED) == ("running", "done", "error", "stuck",
                                "cancelled", "superseded")
    assert RUN_RUNNING not in TERMINAL_RUN_STATUSES
    assert TERMINAL_RUN_STATUSES == {
        "done", "error", "stuck", "cancelled", "superseded"}


def test_runtime_tool_names():
    assert SPAWN_TOOL_NAME == "spawn_subagent"
    assert RUNTIME_TOOL_NAMES == {"spawn_subagent", "find_tools", "load_tools"}
    assert DIRECT_DISCLOSE_THRESHOLD == 8 and LOOP_DETECTION_N == 3


def test_budget_defaults():
    b = RunBudget()
    assert (b.max_steps, b.max_wall_seconds, b.max_subagents,
            b.max_active_tools, b.max_subagent_result_chars) == (
                8, 240.0, 2, 8, 4000)


def test_clamp_child_budget_clamps_wall_clock_and_zeroes_spawn():
    child = clamp_child_budget(RunBudget(), parent_remaining_seconds=30.0)
    assert child.max_wall_seconds == 30.0      # min(240, 30)
    assert child.max_subagents == 0            # depth 1: children never spawn
    assert child.max_steps == 8                # steps are per-run, not clamped


def test_clamp_child_budget_floors_at_one_second():
    child = clamp_child_budget(RunBudget(), parent_remaining_seconds=-5.0)
    assert child.max_wall_seconds == 1.0


def test_models_construct_and_are_frozen_where_stated():
    entry = ToolCatalogEntry(id="builtin:x", name="x",
                             one_line_description="d", source="builtin")
    schema = ToolSchema(id="builtin:x", name="x", description="d",
                        parameters={"type": "object"})
    call = ToolCall(name="x", args={"a": 1})
    result = ToolResult(ok=True, content="42")
    turn = ModelTurn(text="hi", tool_calls=(call,))
    cfg = AgentConfig(model="m", system_prompt="s", allowed_tools=("x",))
    step = AgentStep(index=0, kind="model", summary="hi")
    outcome = RunOutcome(status=RUN_DONE, steps=[step], final_text="hi")
    assert turn.tool_calls[0].args == {"a": 1}
    assert cfg.budget.max_steps == 8 and outcome.subagents_spawned == 0
    for frozen in (entry, schema, call, result, turn, cfg):
        assert dataclasses.fields(frozen)  # constructed fine
        try:
            object.__setattr__  # noqa: B018 — presence check only
            frozen.__class__.__dataclass_params__.frozen
        except AttributeError:  # pragma: no cover
            pass
        assert frozen.__dataclass_params__.frozen is True


def test_pure_module_has_no_forbidden_imports():
    import tldw_chatbook.Agents.agent_models as mod
    src = open(mod.__file__, encoding="utf-8").read()
    for forbidden in ("textual", "sqlite3", "tldw_chatbook.DB",
                      "tldw_chatbook.app", "httpx", "requests"):
        assert forbidden not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_agent_models.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Agents'`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Agents/agent_models.py
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
    """One provider response: raw text plus any native tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


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
```

Also create empty `tldw_chatbook/Agents/__init__.py` and `Tests/Agents/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Agents/test_agent_models.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/__init__.py tldw_chatbook/Agents/agent_models.py Tests/Agents/__init__.py Tests/Agents/test_agent_models.py
git commit -m "feat(agents): pure runtime models and constants

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Fence protocol — parser, stream sniff, protocol renderer (in `agent_runtime.py`)

**Files:**
- Create: `tldw_chatbook/Agents/agent_runtime.py` (protocol half; Task 3 adds the loop to the same file)
- Test: `Tests/Agents/test_tool_protocol.py`

**Interfaces:**
- Consumes: `ToolCall`, `ToolSchema` from Task 1.
- Produces: `FENCE_OPEN = "```tool_call"`; `parse_fenced_tool_call(text: str) -> ToolCall | None` (leading-fence only); `split_visible_text_and_tool_call(text: str) -> tuple[str, ToolCall | None]` (mid-stream fence → visible prefix + call); `STREAM_TOOL_CALL/STREAM_TEXT/STREAM_UNDECIDED` + `stream_prefix_verdict(prefix: str) -> str`; `render_tool_protocol(schemas: list[ToolSchema]) -> str` (the system-prompt section: schemas as JSON + the fence-FIRST instruction).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Agents/test_tool_protocol.py
"""Fence-first text protocol: parse, split, sniff, render."""
import json

from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Agents.agent_runtime import (
    FENCE_OPEN, STREAM_TEXT, STREAM_TOOL_CALL, STREAM_UNDECIDED,
    parse_fenced_tool_call, render_tool_protocol,
    split_visible_text_and_tool_call, stream_prefix_verdict,
)

GOOD = '```tool_call\n{"name": "calculator", "arguments": {"expression": "6*7"}}\n```'


def test_parse_leading_fence():
    call = parse_fenced_tool_call(GOOD)
    assert call is not None
    assert call.name == "calculator" and call.args == {"expression": "6*7"}


def test_parse_allows_leading_whitespace_only():
    assert parse_fenced_tool_call("\n  " + GOOD) is not None
    assert parse_fenced_tool_call("Sure, running it:\n" + GOOD) is None


def test_parse_defensive_on_malformed_json():
    assert parse_fenced_tool_call('```tool_call\n{"name": broken\n```') is None
    assert parse_fenced_tool_call('```tool_call\n"just a string"\n```') is None
    assert parse_fenced_tool_call('```tool_call\n{"arguments": {}}\n```') is None
    assert parse_fenced_tool_call('```tool_call\n{"name": "x", "arguments": []}\n```') is None
    assert parse_fenced_tool_call("```tool_call\n{unclosed") is None
    assert parse_fenced_tool_call("no fence at all") is None


def test_split_mid_stream_fence_truncates_and_converts():
    text = "Let me compute that.\n" + GOOD
    visible, call = split_visible_text_and_tool_call(text)
    assert visible == "Let me compute that."
    assert call is not None and call.name == "calculator"


def test_split_no_fence_returns_text_unchanged():
    visible, call = split_visible_text_and_tool_call("plain answer")
    assert visible == "plain answer" and call is None


def test_split_malformed_fence_stays_visible_text():
    text = "answer ```tool_call\n{broken"
    visible, call = split_visible_text_and_tool_call(text)
    assert call is None and visible == text


def test_stream_sniff_verdicts():
    assert stream_prefix_verdict("") == STREAM_UNDECIDED
    assert stream_prefix_verdict("  \n") == STREAM_UNDECIDED
    assert stream_prefix_verdict("``") == STREAM_UNDECIDED          # fence prefix
    assert stream_prefix_verdict("```tool") == STREAM_UNDECIDED     # fence prefix
    assert stream_prefix_verdict(FENCE_OPEN) == STREAM_TOOL_CALL
    assert stream_prefix_verdict(FENCE_OPEN + "\n{") == STREAM_TOOL_CALL
    assert stream_prefix_verdict("Tokyo is") == STREAM_TEXT
    assert stream_prefix_verdict("```python\n") == STREAM_TEXT      # other fence


def test_render_tool_protocol_contains_schemas_and_fence_first_rule():
    schema = ToolSchema(id="builtin:calculator", name="calculator",
                        description="Evaluate math",
                        parameters={"type": "object", "properties": {
                            "expression": {"type": "string"}}})
    rendered = render_tool_protocol([schema])
    assert "calculator" in rendered and "Evaluate math" in rendered
    assert FENCE_OPEN in rendered
    # The fence-FIRST requirement must be stated explicitly.
    assert "first" in rendered.lower()
    # Schemas must be embedded as valid JSON.
    assert json.dumps(schema.parameters) in rendered or "expression" in rendered


def test_render_empty_schema_list_is_answer_directly():
    rendered = render_tool_protocol([])
    assert rendered == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Agents/test_tool_protocol.py -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` on `agent_runtime`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Agents/agent_runtime.py
"""Pure agent control loop + fence-first text tool protocol.

No Textual, app, DB, or I/O imports. Task 2 adds the protocol; Task 3
appends the loop below it.
"""
from __future__ import annotations

import json

from .agent_models import ToolCall, ToolSchema

FENCE_OPEN = "```tool_call"
_FENCE_CLOSE = "```"

STREAM_TOOL_CALL = "tool_call"
STREAM_TEXT = "text"
STREAM_UNDECIDED = "undecided"


def parse_fenced_tool_call(text: str) -> ToolCall | None:
    """Parse a response whose FIRST non-whitespace content is a tool fence.

    Returns None for anything malformed — never raises.
    """
    stripped = text.lstrip()
    if not stripped.startswith(FENCE_OPEN):
        return None
    after = stripped[len(FENCE_OPEN):]
    newline = after.find("\n")
    if newline == -1:
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
    """
    idx = text.find(FENCE_OPEN)
    if idx == -1:
        return text, None
    call = parse_fenced_tool_call(text[idx:])
    if call is None:
        return text, None
    return text[:idx].rstrip(), call


def stream_prefix_verdict(prefix: str) -> str:
    """Sniff a stream's first tokens: tool_call, text, or undecided."""
    stripped = prefix.lstrip()
    if not stripped:
        return STREAM_UNDECIDED
    if stripped.startswith(FENCE_OPEN):
        return STREAM_TOOL_CALL
    if FENCE_OPEN.startswith(stripped):
        return STREAM_UNDECIDED
    return STREAM_TEXT


def render_tool_protocol(schemas: list[ToolSchema]) -> str:
    """Render the tool-protocol system-prompt section.

    Empty schema list → empty string (no protocol section: answer directly).
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Agents/test_tool_protocol.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_tool_protocol.py
git commit -m "feat(agents): fence-first tool protocol — parse, split, sniff, render

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: The pure control loop (`run_agent_loop` in `agent_runtime.py`)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (append below the protocol code)
- Test: `Tests/Agents/test_agent_runtime.py`

**Interfaces:**
- Consumes: Task 1 models + Task 2 protocol functions.
- Produces: `LoopDeps` dataclass with callables `call_model(messages: list[dict], active_schemas: tuple[ToolSchema, ...]) -> ModelTurn`, `invoke_tool(call: ToolCall) -> ToolResult`, `spawn(task: str) -> ToolResult`, `find_tools(query: str) -> list[ToolCatalogEntry]`, `load_schemas(ids: list[str]) -> list[ToolSchema]`, `should_cancel() -> bool`, `clock() -> float`; and `run_agent_loop(config: AgentConfig, initial_messages: list[dict], active_schemas: list[ToolSchema], deps: LoopDeps) -> RunOutcome`. Message-history convention (text protocol regardless of transport): each model turn appends `{"role": "assistant", "content": <raw turn text>}`; each tool result appends `{"role": "user", "content": f"Tool result for {name}: {content_or_error}"}`.

**Semantics to implement (from the spec — encode exactly):**
- Step budget check BEFORE each model call: `len(steps) >= budget.max_steps` → `stuck`. Wall-clock check there too: `clock() - started > budget.max_wall_seconds` → `stuck`.
- Cancel check at every step boundary (before each model call and before each tool execution) → `cancelled`.
- A turn's calls: native `turn.tool_calls` win; else `split_visible_text_and_tool_call(turn.text)`; neither → `done` with `final_text=turn.text`.
- Runtime tools handled by the loop itself: `spawn_subagent` (respecting `budget.max_subagents`, incrementing `subagents_spawned`, step kind `STEP_SPAWN`), `find_tools` (entries rendered `id — name: one_line_description` per line, or `"No matching tools."`), `load_tools` (extends the active set, silently capped at `budget.max_active_tools`, result names what loaded). Everything else → `deps.invoke_tool`.
- Loop detection: consecutive identical `(name, json.dumps(args, sort_keys=True))` reaching `LOOP_DETECTION_N` → append a `STEP_ERROR` step and return `stuck`.
- A failed tool (`ToolResult.ok is False`) is NOT fatal: record the error in the `STEP_TOOL_RESULT` step and feed `Tool result for {name}: ERROR: {error}` back to the model.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Agents/test_agent_runtime.py
"""Pure loop tests with deterministic fake callables."""
import json

from tldw_chatbook.Agents.agent_models import (
    LOOP_DETECTION_N, RUN_CANCELLED, RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME,
    STEP_SPAWN, STEP_TOOL_RESULT, AgentConfig, ModelTurn, RunBudget,
    ToolCall, ToolCatalogEntry, ToolResult, ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop

CALC = ToolSchema(id="builtin:calculator", name="calculator",
                  description="math", parameters={"type": "object"})


def fence(name, args):
    return f'```tool_call\n{json.dumps({"name": name, "arguments": args})}\n```'


def make_deps(turns, *, invoke=None, spawn=None, cancel=None, clock=None):
    """Deps whose call_model pops scripted ModelTurns."""
    script = list(turns)

    def call_model(messages, active_schemas):
        return script.pop(0)

    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoke or (lambda c: ToolResult(ok=True, content="42")),
        spawn=spawn or (lambda task: ToolResult(ok=True, content="sub done")),
        find_tools=lambda q: [ToolCatalogEntry(
            id="builtin:calculator", name="calculator",
            one_line_description="math", source="builtin")],
        load_schemas=lambda ids: [CALC],
        should_cancel=cancel or (lambda: False),
        clock=clock or (lambda: 0.0),
    )


CFG = AgentConfig(model="m", system_prompt="s",
                  allowed_tools=("calculator", SPAWN_TOOL_NAME))


def run(turns, **kw):
    cfg = kw.pop("config", CFG)
    active = kw.pop("active", [CALC])
    return run_agent_loop(cfg, [{"role": "user", "content": "hi"}],
                          active, make_deps(turns, **kw))


def test_plain_answer_no_tools():
    out = run([ModelTurn(text="Tokyo.")])
    assert out.status == RUN_DONE and out.final_text == "Tokyo."
    assert [s.kind for s in out.steps] == ["model"]


def test_fenced_tool_call_then_answer():
    calls = []
    out = run(
        [ModelTurn(text=fence("calculator", {"expression": "6*7"})),
         ModelTurn(text="It is 42.")],
        invoke=lambda c: calls.append(c) or ToolResult(ok=True, content="42"),
    )
    assert out.status == RUN_DONE and out.final_text == "It is 42."
    assert calls[0].name == "calculator"
    kinds = [s.kind for s in out.steps]
    assert kinds == ["model", "tool_call", "tool_result", "model"]


def test_native_tool_calls_take_precedence_over_text():
    out = run(
        [ModelTurn(text="ignored prose",
                   tool_calls=(ToolCall(name="calculator",
                                        args={"expression": "1"}),)),
         ModelTurn(text="done")])
    assert out.status == RUN_DONE
    assert out.steps[1].tool_name == "calculator"


def test_tool_error_is_not_fatal_and_feeds_back():
    seen = []

    def call_model(messages, active_schemas):
        seen.append(list(messages))
        return (ModelTurn(text=fence("calculator", {"expression": "x"}))
                if len(seen) == 1 else ModelTurn(text="recovered"))

    deps = make_deps([], invoke=lambda c: ToolResult(ok=False, error="boom"))
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [CALC], deps)
    assert out.status == RUN_DONE and out.final_text == "recovered"
    assert "ERROR: boom" in seen[1][-1]["content"]
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert result_steps and "boom" in result_steps[0].result


def test_spawn_result_and_budget():
    tasks = []
    turns = [ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "t1"})),
             ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "t2"})),
             ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "t3"})),
             ModelTurn(text="all done")]
    # 3 spawn rounds + the final answer = 11 steps; default max_steps=8
    # would trip stuck first, so raise it — this test is about spawn budget.
    out = run(turns, spawn=lambda t: tasks.append(t) or ToolResult(
        ok=True, content=f"did {t}"),
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", SPAWN_TOOL_NAME),
                           budget=RunBudget(max_steps=20)))
    assert out.status == RUN_DONE
    assert tasks == ["t1", "t2"]           # max_subagents=2: third refused
    assert out.subagents_spawned == 2
    spawn_steps = [s for s in out.steps if s.kind == STEP_SPAWN]
    assert len(spawn_steps) == 2


def test_disclosure_find_then_load_then_invoke():
    turns = [ModelTurn(text=fence("find_tools", {"query": "math"})),
             ModelTurn(text=fence("load_tools",
                                  {"ids": ["builtin:calculator"]})),
             ModelTurn(text=fence("calculator", {"expression": "6*7"})),
             ModelTurn(text="42.")]
    # find + load + invoke + answer = 10 steps; default max_steps=8 would
    # trip stuck first, so raise it — this test is about disclosure flow.
    out = run(turns, active=[],            # nothing pre-disclosed
              config=AgentConfig(model="m", system_prompt="s",
                                 allowed_tools=("calculator",),
                                 budget=RunBudget(max_steps=16)))
    assert out.status == RUN_DONE and out.final_text == "42."


def test_step_budget_trips_to_stuck():
    endless = [ModelTurn(text=fence("calculator", {"expression": str(i)}))
               for i in range(50)]
    out = run(endless, config=AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=RunBudget(max_steps=4)))
    assert out.status == RUN_STUCK


def test_wall_clock_budget_trips_to_stuck():
    ticker = iter([0.0, 500.0, 1000.0, 1500.0])
    out = run([ModelTurn(text="never reached")],
              clock=lambda: next(ticker))
    assert out.status == RUN_STUCK


def test_identical_consecutive_calls_trip_loop_detection():
    same = ModelTurn(text=fence("calculator", {"expression": "6*7"}))
    out = run([same] * (LOOP_DETECTION_N + 1) + [ModelTurn(text="x")],
              config=AgentConfig(model="m", system_prompt="s",
                                 allowed_tools=("calculator",),
                                 budget=RunBudget(max_steps=50)))
    assert out.status == RUN_STUCK
    assert out.steps[-1].kind == "error"


def test_same_tool_different_args_is_not_stuck():
    turns = [ModelTurn(text=fence("calculator", {"expression": str(i)}))
             for i in range(3)] + [ModelTurn(text="fine")]
    out = run(turns, config=AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=RunBudget(max_steps=50)))
    assert out.status == RUN_DONE


def test_cancel_lands_at_step_boundary():
    flags = iter([False, True])
    out = run([ModelTurn(text=fence("calculator", {"expression": "1"})),
               ModelTurn(text="never")],
              cancel=lambda: next(flags, True))
    assert out.status == RUN_CANCELLED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Agents/test_agent_runtime.py -q`
Expected: FAIL with `ImportError: cannot import name 'LoopDeps'`

- [ ] **Step 3: Write the implementation (append to `agent_runtime.py`)**

```python
# --- appended below the protocol code in tldw_chatbook/Agents/agent_runtime.py ---

from dataclasses import dataclass
from typing import Callable

from .agent_models import (
    FIND_TOOLS_NAME, LOAD_TOOLS_NAME, LOOP_DETECTION_N, RUN_CANCELLED,
    RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME, STEP_ERROR, STEP_MODEL,
    STEP_SPAWN, STEP_TOOL_CALL, STEP_TOOL_RESULT, AgentConfig, AgentStep,
    ModelTurn, RunOutcome, ToolCatalogEntry, ToolResult,
)


@dataclass
class LoopDeps:
    """Everything impure, injected. The loop itself stays pure."""

    call_model: Callable[[list, tuple], ModelTurn]
    invoke_tool: Callable[..., ToolResult]
    spawn: Callable[[str], ToolResult]
    find_tools: Callable[[str], list]
    load_schemas: Callable[[list], list]
    should_cancel: Callable[[], bool]
    clock: Callable[[], float]


def _catalog_lines(entries: list) -> str:
    if not entries:
        return "No matching tools."
    return "\n".join(
        f"{e.id} — {e.name}: {e.one_line_description}" for e in entries)


def run_agent_loop(config: AgentConfig, initial_messages: list[dict],
                   active_schemas: list, deps: LoopDeps) -> RunOutcome:
    """Drive think → (tool) → observe until done / stuck / cancelled.

    Message convention (transport-independent): assistant turns append
    verbatim; tool results append as user-role
    ``Tool result for {name}: {content}`` lines.
    """
    budget = config.budget
    steps: list[AgentStep] = []
    messages = list(initial_messages)
    active = list(active_schemas)
    started = deps.clock()
    spawned = 0
    last_key: tuple | None = None
    repeat_count = 0

    def add(kind: str, **kw) -> AgentStep:
        step = AgentStep(index=len(steps), kind=kind, **kw)
        steps.append(step)
        return step

    while True:
        if deps.should_cancel():
            return RunOutcome(RUN_CANCELLED, steps,
                              subagents_spawned=spawned)
        if len(steps) >= budget.max_steps:
            add(STEP_ERROR, summary="step budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)
        if deps.clock() - started > budget.max_wall_seconds:
            add(STEP_ERROR, summary="wall-clock budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)

        turn = deps.call_model(messages, tuple(active))
        add(STEP_MODEL, summary=turn.text[:200])

        calls = list(turn.tool_calls)
        if not calls:
            _visible, fenced = split_visible_text_and_tool_call(turn.text)
            if fenced is None:
                return RunOutcome(RUN_DONE, steps, final_text=turn.text,
                                  subagents_spawned=spawned)
            calls = [fenced]
        messages.append({"role": "assistant", "content": turn.text})

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

            if call.name == SPAWN_TOOL_NAME:
                task = str(call.args.get("task", ""))
                if spawned >= budget.max_subagents:
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
                ids = list(call.args.get("ids", []))
                loaded = deps.load_schemas(ids)
                room = budget.max_active_tools - len(active)
                accepted = loaded[:max(room, 0)]
                active.extend(accepted)
                result = ToolResult(ok=True, content="loaded: " + ", ".join(
                    s.name for s in accepted) if accepted else "no room")
            else:
                add(STEP_TOOL_CALL, tool_name=call.name,
                    args=dict(call.args))
                result = deps.invoke_tool(call)

            content = result.content if result.ok else f"ERROR: {result.error}"
            add(STEP_TOOL_RESULT, tool_name=call.name,
                result=content[:2000])
            messages.append({
                "role": "user",
                "content": f"Tool result for {call.name}: {content}"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Agents/test_agent_runtime.py Tests/Agents/test_tool_protocol.py -q`
Expected: PASS (all — check the spawn-budget test carefully: a refused third spawn must NOT increment `subagents_spawned` and must not add a `STEP_SPAWN` step, but MUST still add a `STEP_TOOL_RESULT` carrying the error)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_agent_runtime.py
git commit -m "feat(agents): pure constrained control loop with budgets, spawn, disclosure, cancel

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Tool catalog — `ToolProvider` protocol, registry, builtin provider, disclosure (`tool_catalog.py`)

**Files:**
- Create: `tldw_chatbook/Agents/tool_catalog.py`
- Test: `Tests/Agents/test_tool_catalog.py`

**Interfaces:**
- Consumes: Task 1 models; `tldw_chatbook.Tools.tool_executor.DateTimeTool` / `CalculatorTool` (real classes: properties `name`, `description`, `parameters`; `async execute(**kwargs) -> Dict`).
- Produces: `ToolProvider` (typing.Protocol: `list_catalog() -> list[ToolCatalogEntry]`, `load_schema(tool_id: str) -> ToolSchema`, `invoke(tool_id: str, args: dict) -> ToolResult` — invoke is SYNC in this interface; async tools are bridged inside providers); `ToolCatalogRegistry` with `register_provider(p)`, `list_catalog()`, `find(query) -> list[ToolCatalogEntry]` (case-insensitive substring on name + description), `load_schema(tool_id)`, `invoke_by_name(name, args) -> ToolResult`, `resolve_name(name) -> str | None` (tool name → id); `BuiltinToolProvider` (ids `builtin:<tool.name>`; bridges async `execute` via `asyncio.run`, catching every exception into `ToolResult(ok=False, error=...)`; dict results serialized with `json.dumps`); pseudo-tool schemas `SPAWN_TOOL_SCHEMA`, `FIND_TOOLS_SCHEMA`, `LOAD_TOOLS_SCHEMA`; and `initial_disclosure(registry, budget) -> tuple[list[ToolSchema], bool]` — `(direct-disclosed schemas, offer_find_load)` implementing the ≤`DIRECT_DISCLOSE_THRESHOLD` shortcut.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Agents/test_tool_catalog.py
"""Catalog registry + real builtin tools (no network, no DB)."""
from tldw_chatbook.Agents.agent_models import (
    DIRECT_DISCLOSE_THRESHOLD, FIND_TOOLS_NAME, LOAD_TOOLS_NAME,
    RunBudget, SPAWN_TOOL_NAME, ToolCatalogEntry, ToolResult, ToolSchema,
)
from tldw_chatbook.Agents.tool_catalog import (
    FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA, SPAWN_TOOL_SCHEMA,
    BuiltinToolProvider, ToolCatalogRegistry, initial_disclosure,
)


def registry():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


def test_builtin_catalog_lists_calculator_and_datetime():
    entries = registry().list_catalog()
    names = {e.name for e in entries}
    assert {"calculator", "get_current_datetime"} <= names
    assert all(e.id.startswith("builtin:") for e in entries)
    assert all(e.source == "builtin" for e in entries)


def test_find_matches_name_and_description_case_insensitive():
    reg = registry()
    assert any(e.name == "calculator" for e in reg.find("CALC"))
    assert any(e.name == "get_current_datetime" for e in reg.find("timezone"))
    assert reg.find("no-such-thing-xyz") == []


def test_load_schema_round_trip():
    reg = registry()
    schema = reg.load_schema("builtin:calculator")
    assert isinstance(schema, ToolSchema)
    assert schema.name == "calculator"
    assert schema.parameters.get("type") == "object"


def test_invoke_by_name_executes_real_calculator():
    result = registry().invoke_by_name("calculator", {"expression": "6*7"})
    assert result.ok is True
    assert "42" in result.content


def test_invoke_by_name_unknown_tool_is_error_result():
    result = registry().invoke_by_name("nope", {})
    assert result.ok is False and "nope" in result.error


def test_invoke_captures_tool_exception_as_error_result():
    result = registry().invoke_by_name(
        "get_current_datetime", {"timezone": "Not/AZone"})
    assert result.ok is False
    assert result.error  # message captured, no exception escaped


def test_pseudo_tool_schemas():
    assert SPAWN_TOOL_SCHEMA.name == SPAWN_TOOL_NAME
    assert "task" in SPAWN_TOOL_SCHEMA.parameters["properties"]
    assert FIND_TOOLS_SCHEMA.name == FIND_TOOLS_NAME
    assert LOAD_TOOLS_SCHEMA.name == LOAD_TOOLS_NAME


class FakeBigProvider:
    """A provider with more tools than the threshold."""

    def list_catalog(self):
        return [ToolCatalogEntry(id=f"fake:t{i}", name=f"t{i}",
                                 one_line_description=f"tool {i}",
                                 source="fake")
                for i in range(DIRECT_DISCLOSE_THRESHOLD + 3)]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name=tool_id.split(":")[1],
                          description="fake", parameters={"type": "object"})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content="fake")


def test_initial_disclosure_small_catalog_direct_discloses():
    schemas, offer_find_load = initial_disclosure(registry(), RunBudget())
    assert offer_find_load is False
    assert {s.name for s in schemas} >= {"calculator", "get_current_datetime"}


def test_initial_disclosure_large_catalog_defers_to_find_load():
    reg = registry()
    reg.register_provider(FakeBigProvider())
    schemas, offer_find_load = initial_disclosure(reg, RunBudget())
    assert offer_find_load is True and schemas == []


def test_initial_disclosure_respects_max_active_tools():
    schemas, _ = initial_disclosure(registry(),
                                    RunBudget(max_active_tools=1))
    assert len(schemas) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Agents/test_tool_catalog.py -q`
Expected: FAIL with `ModuleNotFoundError` on `tool_catalog`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Agents/tool_catalog.py
"""ToolProvider capability interface + registry + builtin provider.

This is the plugin seam: MCP (task-201) and Skills (task-200) register as
providers here — the runtime never changes. May import tool_executor
(wrapping it is this module's job); no UI/DB imports.
"""
from __future__ import annotations

import asyncio
import json
from typing import Protocol

from tldw_chatbook.Tools.tool_executor import CalculatorTool, DateTimeTool

from .agent_models import (
    DIRECT_DISCLOSE_THRESHOLD, FIND_TOOLS_NAME, LOAD_TOOLS_NAME,
    RunBudget, SPAWN_TOOL_NAME, ToolCatalogEntry, ToolResult, ToolSchema,
)

SPAWN_TOOL_SCHEMA = ToolSchema(
    id="runtime:spawn_subagent",
    name=SPAWN_TOOL_NAME,
    description=(
        "Delegate a self-contained task to an isolated sub-agent. It sees "
        "only the task text you pass, works on it, and returns a result."),
    parameters={
        "type": "object",
        "properties": {"task": {
            "type": "string",
            "description": "Complete, self-contained task description."}},
        "required": ["task"],
    },
)

FIND_TOOLS_SCHEMA = ToolSchema(
    id="runtime:find_tools",
    name=FIND_TOOLS_NAME,
    description="Search the tool catalog by keyword; returns ids + one-liners.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

LOAD_TOOLS_SCHEMA = ToolSchema(
    id="runtime:load_tools",
    name=LOAD_TOOLS_NAME,
    description="Load full schemas for catalog ids so you can call them.",
    parameters={
        "type": "object",
        "properties": {"ids": {"type": "array",
                               "items": {"type": "string"}}},
        "required": ["ids"],
    },
)


class ToolProvider(Protocol):
    """The capability interface providers implement."""

    def list_catalog(self) -> list[ToolCatalogEntry]: ...

    def load_schema(self, tool_id: str) -> ToolSchema: ...

    def invoke(self, tool_id: str, args: dict) -> ToolResult: ...


class BuiltinToolProvider:
    """Wraps tool_executor's built-in tools behind the provider interface."""

    SOURCE = "builtin"

    def __init__(self) -> None:
        self._tools = {t.name: t for t in (CalculatorTool(), DateTimeTool())}

    def _tool_id(self, name: str) -> str:
        return f"{self.SOURCE}:{name}"

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(id=self._tool_id(t.name), name=t.name,
                             one_line_description=t.description,
                             source=self.SOURCE)
            for t in self._tools.values()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.split(":", 1)[1]
        tool = self._tools[name]
        return ToolSchema(id=tool_id, name=tool.name,
                          description=tool.description,
                          parameters=tool.parameters)

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1]
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown builtin tool: {name}")
        try:
            # Providers bridge async tools; the loop's interface is sync.
            # Safe here: the service runs in a worker thread with no
            # running event loop.
            raw = asyncio.run(tool.execute(**args))
        except Exception as exc:  # noqa: BLE001 — captured, never escapes
            return ToolResult(ok=False, error=str(exc))
        if isinstance(raw, dict) and raw.get("error"):
            return ToolResult(ok=False, error=str(raw["error"]))
        content = json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
        return ToolResult(ok=True, content=content)


class ToolCatalogRegistry:
    """Ordered provider registry: catalog, search, schema, invocation."""

    def __init__(self) -> None:
        self._providers: list[ToolProvider] = []

    def register_provider(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    def list_catalog(self) -> list[ToolCatalogEntry]:
        entries: list[ToolCatalogEntry] = []
        for provider in self._providers:
            entries.extend(provider.list_catalog())
        return entries

    def find(self, query: str) -> list[ToolCatalogEntry]:
        needle = query.strip().lower()
        if not needle:
            return []
        return [e for e in self.list_catalog()
                if needle in e.name.lower()
                or needle in e.one_line_description.lower()]

    def _owner_and_id(self, tool_id: str):
        for provider in self._providers:
            if any(e.id == tool_id for e in provider.list_catalog()):
                return provider
        return None

    def load_schema(self, tool_id: str) -> ToolSchema:
        provider = self._owner_and_id(tool_id)
        if provider is None:
            raise KeyError(f"Unknown tool id: {tool_id}")
        return provider.load_schema(tool_id)

    def resolve_name(self, name: str) -> str | None:
        for entry in self.list_catalog():
            if entry.name == name:
                return entry.id
        return None

    def invoke_by_name(self, name: str, args: dict) -> ToolResult:
        tool_id = self.resolve_name(name)
        if tool_id is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        provider = self._owner_and_id(tool_id)
        return provider.invoke(tool_id, args)


def initial_disclosure(
    registry: ToolCatalogRegistry, budget: RunBudget
) -> tuple[list[ToolSchema], bool]:
    """Small catalog → direct-disclose everything, drop find/load.

    Returns (active schemas, offer_find_load).
    """
    catalog = registry.list_catalog()
    if len(catalog) <= DIRECT_DISCLOSE_THRESHOLD:
        schemas = [registry.load_schema(e.id) for e in catalog]
        return schemas[: budget.max_active_tools], False
    return [], True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Agents/test_tool_catalog.py -q`
Expected: PASS (11 tests). If `test_invoke_captures_tool_exception_as_error_result` fails because `DateTimeTool` returns an error dict instead of raising, the `raw.get("error")` branch covers it — verify the assertion still holds and adjust the test's expectation comment, not the contract (`ok=False` either way).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_tool_catalog.py
git commit -m "feat(agents): ToolProvider seam, registry, builtin provider, direct-disclose

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: The run store (`DB/AgentRuns_DB.py`)

**Files:**
- Create: `tldw_chatbook/DB/AgentRuns_DB.py`
- Test: `Tests/DB/test_agent_runs_db.py`

**Interfaces:**
- Consumes: `tldw_chatbook.DB.base_db.BaseDB` — copy the structure of `tldw_chatbook/DB/Workspace_DB.py` (read it first): `_CURRENT_SCHEMA_VERSION`, `connection()` and `transaction()` contextmanagers over `closing(self._get_connection())` (fresh connection per call — this IS the separate-read-connection rule), `_initialize_schema()` with `executescript`.
- Produces: `AgentRunsDB(db_path, client_id="default")` with `create_run(*, conversation_id: str, agent_kind: str, task: str | None = None, parent_run_id: str | None = None, budget: dict | None = None) -> str` (uuid4-hex id, status `running`, ISO-8601 UTC timestamps); `append_steps(run_id: str, steps: list[dict]) -> None`; `set_status(run_id: str, status: str, result: str | None = None) -> None`; `get_run(run_id: str) -> dict | None` (with `steps` parsed back to a list); `list_runs(conversation_id: str, include_superseded: bool = True) -> list[dict]` (newest first); `count_subagent_runs(conversation_id: str) -> int`; `supersede_run_tree(run_id: str) -> int` (marks the run AND every run whose `parent_run_id` is it as `superseded`; returns rows changed).

- [ ] **Step 1: Write the failing test**

```python
# Tests/DB/test_agent_runs_db.py
"""AgentRunsDB against a real on-disk SQLite file."""
import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "agent_runs.db", client_id="test")


def test_create_and_get_run(db):
    run_id = db.create_run(conversation_id="conv1", agent_kind="primary",
                           budget={"max_steps": 8})
    run = db.get_run(run_id)
    assert run["conversation_id"] == "conv1"
    assert run["agent_kind"] == "primary"
    assert run["status"] == "running"
    assert run["steps"] == [] and run["parent_run_id"] is None
    assert run["budget"] == {"max_steps": 8}
    assert run["created_at"] and run["updated_at"]


def test_get_missing_run_returns_none(db):
    assert db.get_run("nope") is None


def test_append_steps_accumulates_and_parses(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.append_steps(run_id, [{"index": 0, "kind": "model", "summary": "hi"}])
    db.append_steps(run_id, [{"index": 1, "kind": "tool_call",
                              "tool_name": "calculator"}])
    steps = db.get_run(run_id)["steps"]
    assert [s["index"] for s in steps] == [0, 1]
    assert steps[1]["tool_name"] == "calculator"


def test_set_status_and_result(db):
    run_id = db.create_run(conversation_id="c", agent_kind="subagent",
                           task="do x", parent_run_id="p1")
    db.set_status(run_id, "done", result="the answer")
    run = db.get_run(run_id)
    assert run["status"] == "done" and run["result"] == "the answer"
    assert run["task"] == "do x" and run["parent_run_id"] == "p1"


def test_count_subagents_counts_only_subagent_kind(db):
    db.create_run(conversation_id="c", agent_kind="primary")
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    for i in range(3):
        db.create_run(conversation_id="c", agent_kind="subagent",
                      task=f"t{i}", parent_run_id=parent)
    db.create_run(conversation_id="other", agent_kind="subagent", task="x",
                  parent_run_id="zzz")
    assert db.count_subagent_runs("c") == 3


def test_supersede_run_tree_marks_run_and_children(db):
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    child = db.create_run(conversation_id="c", agent_kind="subagent",
                          task="t", parent_run_id=parent)
    other = db.create_run(conversation_id="c", agent_kind="primary")
    changed = db.supersede_run_tree(parent)
    assert changed == 2
    assert db.get_run(parent)["status"] == "superseded"
    assert db.get_run(child)["status"] == "superseded"
    assert db.get_run(other)["status"] == "running"


def test_list_runs_filters_superseded_when_asked(db):
    a = db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="primary")
    db.supersede_run_tree(a)
    assert len(db.list_runs("c")) == 2
    live = db.list_runs("c", include_superseded=False)
    assert len(live) == 1 and live[0]["status"] == "running"


def test_sql_is_parameterized_against_quotes(db):
    run_id = db.create_run(conversation_id="c''; DROP TABLE agent_runs;--",
                           agent_kind="primary", task="a 'quoted' task")
    assert db.get_run(run_id)["task"] == "a 'quoted' task"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/DB/test_agent_runs_db.py -q`
Expected: FAIL with `ModuleNotFoundError` on `AgentRuns_DB`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/DB/AgentRuns_DB.py
"""SQLite persistence for agent run records (primary + sub-agent).

Follows the Workspace_DB pattern: BaseDB, per-call connections (reads get
their own connection automatically), transaction() for writes.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Union

from .base_db import BaseDB


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class AgentRunsDB(BaseDB):
    """Run records for the agent runtime (vertical-slice spec data model)."""

    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: Union[str, Path],
                 client_id: str = "default") -> None:
        super().__init__(db_path, client_id)

    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with closing(self._get_connection()) as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._get_connection()) as conn:
            conn.execute("BEGIN")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def _initialize_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY NOT NULL
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (1);

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    parent_run_id TEXT,
                    agent_kind TEXT NOT NULL,
                    task TEXT,
                    status TEXT NOT NULL,
                    steps TEXT NOT NULL DEFAULT '[]',
                    result TEXT,
                    budget TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
                    ON agent_runs(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_parent
                    ON agent_runs(parent_run_id);
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["steps"] = json.loads(record["steps"] or "[]")
        record["budget"] = (json.loads(record["budget"])
                            if record["budget"] else None)
        return record

    def create_run(self, *, conversation_id: str, agent_kind: str,
                   task: str | None = None, parent_run_id: str | None = None,
                   budget: dict | None = None) -> str:
        run_id = uuid.uuid4().hex
        now = _now_iso()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (id, conversation_id, parent_run_id, agent_kind, task,
                    status, steps, result, budget, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'running', '[]', NULL, ?, ?, ?)""",
                (run_id, conversation_id, parent_run_id, agent_kind, task,
                 json.dumps(budget) if budget is not None else None,
                 now, now),
            )
        return run_id

    def append_steps(self, run_id: str, steps: list[dict]) -> None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT steps FROM agent_runs WHERE id = ?",
                (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run id: {run_id}")
            existing = json.loads(row["steps"] or "[]")
            existing.extend(steps)
            conn.execute(
                "UPDATE agent_runs SET steps = ?, updated_at = ? "
                "WHERE id = ?",
                (json.dumps(existing), _now_iso(), run_id))

    def set_status(self, run_id: str, status: str,
                   result: str | None = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE agent_runs SET status = ?, "
                "result = COALESCE(?, result), updated_at = ? WHERE id = ?",
                (status, result, _now_iso(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?",
                (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(self, conversation_id: str,
                  include_superseded: bool = True) -> list[dict]:
        query = "SELECT * FROM agent_runs WHERE conversation_id = ?"
        params: list = [conversation_id]
        if not include_superseded:
            query += " AND status != 'superseded'"
        query += " ORDER BY created_at DESC, id DESC"
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_subagent_runs(self, conversation_id: str) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM agent_runs "
                "WHERE conversation_id = ? AND agent_kind = 'subagent'",
                (conversation_id,)).fetchone()
        return int(row["n"])

    def supersede_run_tree(self, run_id: str) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE agent_runs SET status = 'superseded', "
                "updated_at = ? WHERE id = ? OR parent_run_id = ?",
                (_now_iso(), run_id, run_id))
            return cursor.rowcount
```

Note: `BaseDB.__init__` calls `_initialize_schema()` — verify by reading `tldw_chatbook/DB/base_db.py` before running; if it does not, call it at the end of `__init__` exactly as `Workspace_DB` does.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/DB/test_agent_runs_db.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py Tests/DB/test_agent_runs_db.py
git commit -m "feat(agents): AgentRunsDB run-record store

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: The service (`agent_service.py`) — wiring, permission gate, spawn, supersede, persistence

**Files:**
- Create: `tldw_chatbook/Agents/agent_service.py`
- Test: `Tests/Agents/test_agent_service.py`

**Interfaces:**
- Consumes: everything above, plus `chat_api_call` (injected as a callable — tests NEVER hit the network; the default in production is `tldw_chatbook.Chat.Chat_Functions.chat_api_call`, imported lazily inside `__init__`'s default so importing the module doesn't drag the app config in).
- Produces: `AgentService(db: AgentRunsDB, registry: ToolCatalogRegistry, chat_call=None, clock=time.monotonic)` with the single public entry point:
  `run_turn(*, conversation_id: str, messages: list[dict], config: AgentConfig, api_endpoint: str, should_cancel=lambda: False, supersede_run_id: str | None = None) -> tuple[str, RunOutcome]` (returns `(primary_run_id, outcome)`).
- Behavior (each item is a test):
  1. **Text protocol always** (llama_cpp's `tools` map is commented out — verified): the system message sent to `chat_call` = `config.system_prompt` + `\n\n` + `render_tool_protocol(runtime pseudo-tools + active set)`. Runtime pseudo-tools included: `SPAWN_TOOL_SCHEMA` always (when `budget.max_subagents > 0`); `FIND_TOOLS_SCHEMA`+`LOAD_TOOLS_SCHEMA` only when `initial_disclosure` returned `offer_find_load=True`. The system message is REBUILT each call from the loop's current `active_schemas` argument (disclosure grows it).
  2. `chat_call` invocation shape: `chat_call(api_endpoint=api_endpoint, messages_payload=[system] + messages, streaming=False, model=config.model)`; response text extracted from `resp["choices"][0]["message"]["content"]` (empty string if missing) → `ModelTurn(text=...)`.
  3. **Permission gate**: `invoke_tool` rejects (as `ToolResult(ok=False, error="Tool not permitted: {name}")`) any call whose name is NOT in `config.allowed_tools` OR not in the currently-disclosed active set (tracked by the service via a mutable active-names set that `load_schemas` extends). Allowed+disclosed → `registry.invoke_by_name`.
  4. **Spawn**: creates a child run record (`agent_kind="subagent"`, `parent_run_id`, `task`); child config = same model, the sub-agent system prompt (`SUBAGENT_SYSTEM_PROMPT` constant: complete the task, report a concise result, no user interaction), `allowed_tools` = parent's minus `spawn_subagent`, budget = `clamp_child_budget(config.budget, parent_remaining_seconds)` where remaining = `config.budget.max_wall_seconds - (clock() - turn_start)`; child messages = `[{"role": "user", "content": task}]` ONLY (clean context); runs `run_agent_loop` recursively with its own deps; persists child steps/status/result; returns `ToolResult(ok=True, content=<result truncated to max_subagent_result_chars with "\n[truncated]" suffix when cut>)`; a child that ends `stuck`/`cancelled`/`error` returns `ToolResult(ok=False, error=f"sub-agent {status}")` with partial text if any.
  5. **Supersede**: when `supersede_run_id` is passed, `db.supersede_run_tree(supersede_run_id)` runs BEFORE the new run record is created.
  6. **Persistence**: primary run record created up front (budget serialized via `dataclasses.asdict`); after the loop, all steps stamped with `created_at=_now_iso()` equivalents (service-side ISO stamps via `datetime.now(timezone.utc)`) and appended (`dataclasses.asdict(step)`); status set from `outcome.status`, `result=outcome.final_text`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Agents/test_agent_service.py
"""Service tests: scripted chat_call (no network) + real AgentRunsDB."""
import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME, AgentConfig, RunBudget,
)
from tldw_chatbook.Agents.agent_service import (
    SUBAGENT_SYSTEM_PROMPT, AgentService,
)
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider, ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def fence(name, args):
    return f'```tool_call\n{json.dumps({"name": name, "arguments": args})}\n```'


def provider_reply(text):
    return {"choices": [{"message": {"content": text}}]}


class ScriptedChat:
    """Returns scripted replies; records every call's kwargs."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return provider_reply(self.replies.pop(0))


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def make_service(db, replies):
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = ScriptedChat(replies)
    return AgentService(db=db, registry=registry, chat_call=chat), chat


CFG = AgentConfig(model="test-model", system_prompt="You are helpful.",
                  allowed_tools=("calculator", "get_current_datetime",
                                 SPAWN_TOOL_NAME))


def test_plain_answer_persists_done_run(db):
    service, chat = make_service(db, ["Tokyo."])
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "capital of Japan?"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE and outcome.final_text == "Tokyo."
    run = db.get_run(run_id)
    assert run["status"] == "done" and run["result"] == "Tokyo."
    assert run["agent_kind"] == "primary"
    assert all(s["created_at"] for s in run["steps"])


def test_system_message_carries_protocol_and_user_prompt(db):
    service, chat = make_service(db, ["hi"])
    service.run_turn(conversation_id="c", messages=[
        {"role": "user", "content": "q"}], config=CFG,
        api_endpoint="llama_cpp")
    call = chat.calls[0]
    assert call["api_endpoint"] == "llama_cpp"
    assert call["streaming"] is False and call["model"] == "test-model"
    system = call["messages_payload"][0]
    assert system["role"] == "system"
    assert "You are helpful." in system["content"]
    assert "```tool_call" in system["content"]          # protocol rendered
    assert "calculator" in system["content"]            # direct-disclosed
    assert SPAWN_TOOL_NAME in system["content"]
    assert "find_tools" not in system["content"]        # small catalog


def test_real_tool_executes_through_gate(db):
    service, chat = make_service(
        db, [fence("calculator", {"expression": "6*7"}), "It is 42."])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "6*7?"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE and outcome.final_text == "It is 42."
    # The tool result really came from CalculatorTool:
    followup = chat.calls[1]["messages_payload"]
    assert any("42" in m["content"] for m in followup
               if m["role"] == "user" and "Tool result" in m["content"])


def test_permission_gate_blocks_disallowed_tool(db):
    narrow = AgentConfig(model="m", system_prompt="s",
                         allowed_tools=("get_current_datetime",))
    service, _ = make_service(
        db, [fence("calculator", {"expression": "1"}), "gave up"])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=narrow, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert results and "not permitted" in results[0]["result"]


def test_spawn_creates_linked_child_with_clean_context(db):
    service, chat = make_service(db, [
        fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),   # parent turn 1
        "sub answer: 42",                                  # CHILD turn 1
        "The sub-agent says 42.",                          # parent turn 2
    ])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[
            {"role": "user", "content": "delegate this"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE and outcome.subagents_spawned == 1
    runs = db.list_runs("c")
    child = next(r for r in runs if r["agent_kind"] == "subagent")
    assert child["parent_run_id"] == run_id
    assert child["task"] == "compute 6*7"
    assert child["status"] == "done" and child["result"] == "sub answer: 42"
    # Clean context: the child's provider call saw ONLY its task + its own
    # system prompt — never the parent's transcript.
    child_call = chat.calls[1]["messages_payload"]
    assert child_call[0]["role"] == "system"
    assert SUBAGENT_SYSTEM_PROMPT.split(".")[0] in child_call[0]["content"]
    assert child_call[1] == {"role": "user", "content": "compute 6*7"}
    assert not any("delegate this" in m["content"] for m in child_call)
    assert db.count_subagent_runs("c") == 1


def test_subagent_result_is_capped(db):
    long_answer = "x" * 10000
    service, _ = make_service(db, [
        fence(SPAWN_TOOL_NAME, {"task": "t"}), long_answer, "done"])
    tight = AgentConfig(model="m", system_prompt="s",
                        allowed_tools=(SPAWN_TOOL_NAME,),
                        budget=RunBudget(max_subagent_result_chars=100))
    _, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=tight, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    run = db.list_runs("c", include_superseded=False)
    parent = next(r for r in run if r["agent_kind"] == "primary")
    capped = [s for s in parent["steps"] if s["kind"] == "tool_result"][0]
    assert "[truncated]" in capped["result"]
    assert len(capped["result"]) < 300


def test_child_cannot_spawn(db):
    service, _ = make_service(db, [
        fence(SPAWN_TOOL_NAME, {"task": "t"}),          # parent spawns
        fence(SPAWN_TOOL_NAME, {"task": "nested"}),     # CHILD tries to spawn
        "child recovered",                              # child answers
        "parent done",
    ])
    _, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c") == 1             # no grandchildren


def test_supersede_marks_old_tree_before_new_run(db):
    service, _ = make_service(db, ["first answer"])
    old_id, _ = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp")
    service2, _ = make_service(db, ["second answer"])
    new_id, _ = service2.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp", supersede_run_id=old_id)
    assert db.get_run(old_id)["status"] == "superseded"
    assert db.get_run(new_id)["status"] == "done"


def test_stuck_run_persists_stuck_status(db):
    replies = [fence("calculator", {"expression": "1"})] * 10
    service, _ = make_service(db, replies)
    tight = AgentConfig(model="m", system_prompt="s",
                        allowed_tools=("calculator",),
                        budget=RunBudget(max_steps=3))
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=tight, api_endpoint="llama_cpp")
    assert outcome.status == RUN_STUCK
    assert db.get_run(run_id)["status"] == "stuck"


def test_provider_exception_persists_error_status(db):
    def exploding_chat(**kwargs):
        raise RuntimeError("connection refused")

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(db=db, registry=registry,
                           chat_call=exploding_chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == "error"
    run = db.get_run(run_id)
    assert run["status"] == "error"
    assert any("connection refused" in (s.get("summary") or "")
               for s in run["steps"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest Tests/Agents/test_agent_service.py -q`
Expected: FAIL with `ModuleNotFoundError` on `agent_service`

- [ ] **Step 3: Write the implementation**

```python
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
        disclosed_names = {schema.name for schema in active}
        runtime_schemas = []
        if config.budget.max_subagents > 0:
            runtime_schemas.append(SPAWN_TOOL_SCHEMA)
        if offer_find_load:
            runtime_schemas.extend([FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA])

        def load_schemas(ids: list):
            schemas = []
            for tool_id in ids:
                try:
                    schema = self.registry.load_schema(str(tool_id))
                except KeyError:
                    continue
                schemas.append(schema)
                disclosed_names.add(schema.name)
            return schemas

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
            find_tools=self.registry.find,
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
        """Run one primary-agent turn; returns (run_id, outcome)."""
        if supersede_run_id:
            self.db.supersede_run_tree(supersede_run_id)
        return self._run_one(
            conversation_id=conversation_id, messages=messages,
            config=config, api_endpoint=api_endpoint,
            should_cancel=should_cancel, agent_kind=AGENT_KIND_PRIMARY,
            task=None, parent_run_id=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest Tests/Agents/test_agent_service.py -q`
Expected: PASS (11 tests). Watch `test_spawn_creates_linked_child_with_clean_context`: the child's chat call is `chat.calls[1]` because sub-agents run synchronously inside the parent's tool dispatch.

- [ ] **Step 5: Run the whole Plan-A suite + commit**

Run: `$PY -m pytest Tests/Agents Tests/DB/test_agent_runs_db.py -q`
Expected: ALL PASS

```bash
git add tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service.py
git commit -m "feat(agents): AgentService — provider wiring, permission gate, spawn, supersede

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Plan-A exit criteria

- All six task suites green under the venv python.
- No module outside `agent_service.py` imports `chat_api_call` or any DB; `agent_models.py`/`agent_runtime.py` import nothing but stdlib + each other (the Task-1 purity test enforces this for models; reviewers check runtime).
- The engine is fully drivable headlessly: `AgentService.run_turn` with a scripted `chat_call` produces persisted primary + sub-agent run records — this is exactly the surface Plan B (Console UI: send-path integration, rail inspector, `[N Sub-Agents]` node, Stop, live gate) builds on.
- Plan B is written AFTER Plan A ships, so it can target whatever the PR #620 rebase reality is (`chat_screen.py` conflicts) — per the spec's cross-project coordination section.
