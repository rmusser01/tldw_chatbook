# Agent runtime — vertical slice — design

Date: 2026-07-12 (approved in-session by user)
Status: Approved design; implementation plan to follow.

Sub-project 1 of an agent-runtime **program**. The program's vision (user-stated): a small,
constrained, eventually self-extending agent — in the spirit of pi / cheetahclaws — arranged as
**workspace → conversation → agent → sub-agent**, that a user can spawn and interact with, with
Skills and MCP tools as plugins. This spec covers **only the first vertical slice**: the smallest
working spine through that hierarchy. Later sub-projects (saved agent templates, multi-agent rooms,
self-extension, MCP/Skills as plugins, ACP as an agent kind) each get their own spec.

## Purpose

Prove the spine end-to-end at minimum breadth, full depth: one workspace → one Console conversation
→ one **primary agent** that runs a constrained tool-calling loop and can spawn **ephemeral
sub-agents** (each an isolated, inspectable session that reports a result back). Real tool execution
against built-in tools; progressive tool disclosure; a permission gate; live-cancellable runs; and
the Console UI to watch and drill into the sub-agent tree. Everything is built so Skills (task-200)
and MCP (task-201) later register as tool providers through one interface defined here.

## Decisions locked with the user

- **Entity model:** one **primary agent per conversation** (its own model + system prompt + tool
  set); the primary spawns **ephemeral sub-agents** that run an isolated task and report a result
  back into the transcript (Claude-Code Task-tool model). Not a multi-agent room; not an
  agent-owns-conversations model.
- **Surface:** the existing **Console**. Left conversation tree gains a tiered `[N Sub-Agents]`
  node per conversation; the right status rail becomes an **agent inspector** (agent actions +
  sub-agent list + click-through into any sub-agent's session, with stuck/error surfaced).
- **Runtime:** a new **internal constrained loop** reusing the app's provider path (`chat_api_call`)
  and `tool_executor`. **Not** ACP. (ACP — driving an *external* agent process — stays a possible
  future **alternative agent engine** surfaced through the same Console run/inspector UI; that is a
  different extension axis from the tool-provider interface, and out of scope here.)
- **Sub-agent session storage:** **dedicated run-record store** (a new `AgentRunsDB`), not
  overloaded onto conversations/messages.
- **Progressive tool disclosure** is a first-class runtime property (see below), built as a real
  mechanism in the slice even with only 2–3 built-in tools.
- **Deferred to named follow-ups** (not this slice): saved/reusable agent templates; multi-agent
  rooms; agent self-extension; ACP agent kind; MCP tools as a provider (task-201); Skills as a
  provider + `/skill-name` triggering a skill inside a run (task-200); parallel sub-agents; nested
  sub-agents beyond depth 1; tool-result streaming.

## Architecture — isolated, testable units

- `Agents/agent_models.py` — **pure** dataclasses, no Textual/DB/IO: `AgentConfig`,
  `ToolCatalogEntry`, `AgentStep`, `SubAgentSession`, `RunBudget`, `RunStatus`
  (`running|done|error|stuck|cancelled`).
- `Agents/agent_runtime.py` — the **pure control loop**. Given injected callables
  (`call_model`, `list_catalog`, `load_schema`, `invoke_tool`, `spawn`, `should_cancel`), it drives
  *think → (disclose/call tools) → observe → repeat* until a terminal state or a budget trip. No
  Textual/app/DB imports; fully unit-testable with fakes.
- `Agents/agent_service.py` — wires the runtime to the **real** world: the provider (`chat_api_call`),
  the tool providers + permission gate, the fallback tool-protocol parser, fan-out/budget
  enforcement, spawn (depth-limited), and persistence (`AgentRunsDB`).
- `Agents/tool_catalog.py` — the **capability interface** + registry: a `ToolProvider` protocol and
  the catalog/disclosure/invocation plumbing (the seam MCP/Skills register through).
- `DB/AgentRuns_DB.py` — the dedicated run store.
- Console integration: `chat_screen.py` (start/stream/cancel an agent run; inline tool/spawn
  markers), a new right-rail **agent inspector** widget, and the left-tree `[N Sub-Agents]` node.

## The capability / tool interface (the plugin seam)

One interface owns *what tools exist and how they're disclosed and invoked*; providers own *where
tools come from*.

```
ToolCatalogEntry:  id, name, one_line_description, source        # cheap to list
ToolProvider (protocol):
    list_catalog() -> list[ToolCatalogEntry]                     # names + one-liners only
    load_schema(tool_id) -> ToolSchema                           # full JSON schema, on demand
    invoke(tool_id, args) -> ToolResult                          # actual execution
```

The runtime holds an ordered list of registered providers. For the slice there is exactly one:
`BuiltinToolProvider` (wraps `tool_executor`'s `calculator`/`datetime` + the runtime's own
`spawn_subagent`). **MCP** (task-201) and **Skills** (task-200) each become a `ToolProvider` — no
runtime changes; this is the socket, they are the plugs. Invocation always routes through the
**permission gate** (below).

## Progressive tool disclosure

The model never sees the whole catalog. Each turn the provider receives only:
- the small **always-on core**: `find_tools(query)`, `load_tools(ids)`, `spawn_subagent(task)`;
- plus the **active set** — schemas the agent has explicitly loaded this run.

Flow: the agent calls `find_tools("math")` → gets matching `ToolCatalogEntry` name+one-liners →
calls `load_tools([...])` → those schemas enter the active set and are offered on subsequent turns →
the agent calls them. This keeps context lean (the "constrained agent" property) and is the exact
attach point for large MCP/Skills catalogs later. Mirrors this harness's own ToolSearch. A run's
active set is capped (budget) so disclosure can't balloon context.

## Runtime loop, tool-calling, and the fallback protocol

Primary agent = model (defaults to the Console session's model) + a default agent system prompt (a
small prompt telling the agent how to use `find_tools`/`load_tools`/`spawn_subagent` and to answer
directly when no tool is needed) + the core tools. The loop:

1. Call the provider with the current message history + active tool schemas.
2. If the response contains tool calls → for each: permission-gate → route to the owning provider →
   append a `tool_result` step → loop. `find_tools`/`load_tools`/`spawn_subagent` are handled by the
   runtime itself.
3. If no tool calls → the response text is the **final answer**; run → `done`.
4. Budgets bound every run.

**Tool-calling transport** (robustness — local models are unreliable at native function-calling):
the runtime supports **native tool-calls** where the provider advertises them, and a **text-protocol
fallback** otherwise — tool schemas are rendered into the context with an instruction to emit a
fenced ` ```tool_call {json}``` ` block, which the runtime parses. If neither native calls nor a
parseable block appear, the output is treated as a final answer (graceful degradation — the slice
still produces a response on a tool-incapable model).

## Sub-agents — isolation, spawn, budgets

- **Spawn is a tool** (`spawn_subagent(task, ...)`). When the primary calls it, the service starts a
  **child run of the same engine** with a **clean, isolated context** — the sub-agent sees only its
  task string and its own tools, **not** the parent transcript (context hygiene is the point of
  delegation). It returns a **result string** that becomes the parent's `tool_result`.
- **Depth 1 only** in the slice: a sub-agent's tool set **excludes** `spawn_subagent`.
- **Sequential** execution (parallel sub-agents are a follow-up).
- **Fan-out + budgets:** `RunBudget(max_steps, max_wall_seconds, max_subagents, max_active_tools)`
  bounds every run; the parent enforces `max_subagents` per conversation-run.

## Stuck, error, and cancellation

- **stuck** := budget trip (max steps or wall-clock) **OR** N identical consecutive tool calls
  (cheap loop detection, N=3 default). Surfaced in the rail + an inline marker; the run stops.
- **error** := an unrecoverable provider/tool error; captured into the step log; a failed sub-agent
  spawn returns an error result to the parent and never corrupts the parent run.
- **cancel** := a hard user **Stop** that cancels the whole tree (primary + any running sub-agent)
  via `should_cancel`; the run runs as a Textual background worker so the UI never blocks; state is
  persisted as `cancelled`.

## Permission gate

Every tool invocation passes an allow-list checkpoint in `agent_service` **before** execution —
even for the safe built-ins, so the checkpoint is not retrofitted. Only tools that are both
**disclosed** (in the active set) **and allowed** for the agent are callable. This is where the
Skills trust model and MCP per-workspace permissions plug in later.

## Data model — `AgentRunsDB` (dedicated store; option A)

Its own small DB (following the per-subsystem DB pattern), keyed by conversation. One **run record
per agent**, primary and sub-agent alike:

- `agent_runs`: `id`, `conversation_id`, `parent_run_id` (NULL for the primary), `agent_kind`
  (`primary|subagent`), `task` (sub-agent's task; NULL for primary), `status`, `steps` (JSON list of
  `AgentStep`), `result` (sub-agent result text), `budget` (JSON), `created_at`, `updated_at`.
- `AgentStep`: `index`, `kind` (`model|tool_call|tool_result|spawn|error`), `summary`, `tool_name`,
  `args`, `result`, timestamps.
- `[N Sub-Agents]` count = rows where `conversation_id = X AND agent_kind = 'subagent'`.
- Drill-in = load a sub-agent run record and render its `steps`.

The primary agent's **user-facing** messages remain normal Console conversation messages; its
**internal** tool/spawn steps live in its run record (and as compact inline markers).

## Console UI

- **Left tree:** under a conversation row, a dim, markup-escaped `[N Sub-Agents]` child node, shown
  when N>0; expandable to sub-agent rows (task + status glyph).
- **Right rail — Agent inspector:** the active agent's status + recent action log (step tail); a
  sub-agent list with per-item status (`● running · ✓ done · ⚠ stuck · ✗ error`); clicking a
  sub-agent opens its **session view** (its step log) in the rail with a Back affordance; stuck/error
  visibly flagged.
- **Inline markers:** tool calls and spawns drop compact transcript markers (reusing the existing
  tool-message widget pattern) so the main flow stays readable without the rail.
- **Stop:** a control that cancels the whole run tree.

## Error handling (user-facing)

- Tool-incapable model → the loop still answers (fallback → plain answer) and the rail notes "no
  tools available for this model."
- Provider/DB failure mid-run → honest status in the rail; partial steps preserved; never a silent
  hang (budgets + cancel guarantee termination).
- All agent/sub-agent/task/tool text rendered into labels/rows/markers is markup-escaped.

## Testing

- **Pure (`agent_runtime`):** fake `call_model`/tool/`spawn`/`should_cancel` callables — plain
  answer (no tools); tool-call loop; disclosure (`find_tools`→`load_tools`→invoke); spawn→result;
  budget→`stuck`; loop-detection→`stuck`; tool-error captured; cancel mid-run. Fallback-protocol
  parser: native calls vs text `tool_call` block vs neither.
- **Service (real `AgentRunsDB`):** run persisted; sub-agent run linked via `parent_run_id`; status
  transitions; permission gate blocks a non-allowed/undisclosed tool; fan-out cap enforced.
- **UI (real App + `run_test`):** `[N Sub-Agents]` count node; rail shows status + sub-agent list;
  drill-in renders a sub-agent session; Stop cancels; escaped names.
- **Live gate (served, 2050×1240):** a real agent conversation where the primary calls a built-in
  tool **and** spawns a sub-agent that returns a result, rail showing the tree; captured. Per the
  established capture recipe.

## Out of scope (named follow-ups)

Saved/reusable agent templates; multi-agent rooms; agent self-extension; ACP as an agent kind
(through the same provider/runtime interface); **MCP tools as a `ToolProvider` (task-201 — parallel
session; targets the capability interface defined here)**; **Skills as a `ToolProvider` +
`/skill-name` triggering a skill inside a run (task-200)**; parallel sub-agent execution; nested
sub-agents beyond depth 1; tool-result/partial streaming in the inspector; **ACP as an alternative
agent engine** (an external-process agent behind the same Console run/inspector surface — a
different axis from tool providers).
