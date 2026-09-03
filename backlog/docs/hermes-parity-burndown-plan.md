# Hermes parity: burn-down plan

Execution ordering for the 56 tasks filed from the 2026-08-31 parity report
(**TASK-25900–25914**, **TASK-26001–26040**, and **TASK-28227**—formerly
TASK-26000). Companion to
[hermes-parity-deferred-items.md](hermes-parity-deferred-items.md), which covers what was *not* filed, and to
`qa/hermes-parity-2026-08-31/report.md`, which carries the evidence.

The grouping is derived from **file collisions**, not from theme. Two tasks that edit the same file cannot run
in parallel no matter how unrelated they sound, and two tasks in different subsystems can run concurrently even
when they look similar. The collision map below is the input; the lanes are the output.

> **Re-derive before relying on this.** Collisions shift as work lands — a task that closes may free a file, and
> a refactor may create a new hotspot. The map was computed from the file paths cited in each task's description
> at filing time. Recompute it if the plan feels wrong rather than trusting this snapshot.

---

## The constraint: files cited by more than one task

| Tasks | File | Task IDs |
|---:|---|---|
| 8 | `Agents/agent_runtime.py` | 25901, 25903, 25913, 26000, 26001, 26002, 26005, 26010 |
| 5 | `Chat/console_chat_controller.py` | 25910, 26000, 26004, 26010, 26017 |
| 4 | `Chat/console_context_compaction.py` | 25910, 26016, 26017, 26018 |
| 4 | `config.py` | 26036, 26037, 26038, 26040 |
| 3 | `MCP/local_store.py` | 25900, 26013, 26032 |
| 3 | `MCP/client.py` | 25900, 26029, 26030 |
| 3 | `UI/Screens/settings_screen.py` | 25906, 26038, 26039 |
| 3 | `Chat/console_agent_bridge.py` | 25913, 26002, 26003 |
| 2 | `Chat/Chat_Functions.py` | 25902, 26024 |
| 2 | `Agents/agent_service.py` | 25903, 25913 |
| 2 | `Tools/raw_cli_executor.py` | 25905, 26006 |
| 2 | `Chat/console_command_suggestions.py` | 25908, 26020 |
| 2 | `Chat/console_command_grammar.py` | 25908, 25909 |
| 2 | `app.py` | 25908, 25914 |
| 2 | `Chat/console_history_budget.py` | 25911, 25912 |
| 2 | `Agents/agent_models.py` | 25913, 26007 |
| 2 | `Agents/tool_catalog.py` | 26007, 26008 |
| 2 | `Widgets/Chat_Widgets/chat_approval_card.py` | 26009, 26012 |
| 2 | `Agents/mcp_tool_provider.py` | 26011, 26030 |
| 2 | `Chat/provider_usage.py` | 26014, 26015 |
| 2 | `Scheduling/scheduler/loop.py` | 26025, 26028 |

---

## Eight lanes

Within a lane, work is **serial**. Across lanes it is **parallel-safe**, except for the three cross-lane
collisions called out at the end. One worktree per lane.

### Lane 1 — Agent loop ⚠️ critical path

Ten tasks serialized on `Agents/agent_runtime.py`. This lane is the schedule driver and contains the two
largest interaction changes. Ordered small-to-large so the risky ones land on a warm lane.

`26005` arg coercion → `25913` timeout clamp → `25901` retry → `25902` fallback → `26002` empty-response →
`26001` graceful wrap-up → `26010` post-tool hook → `25903` mid-run steering → `28227` active-turn redirect (formerly `26000`)

**Peels off:** `26003` stall watchdog touches `console_provider_gateway.py` / `console_agent_bridge.py`, not
`agent_runtime.py` — run it beside the lane, not inside it.

### Lane 2 — Context & compaction

- **Serial** (`console_context_compaction.py`): `26016` timeout → `26017` preview → `26018` focus →
  `25910` micro-compaction → `26021` provider-native
- **Parallel** (`console_history_budget.py`, serial with each other): `25911` pruning → `25912` stale-image
- **Parallel**: `26019` context breakdown

### Lane 3 — Tools & permissions

- **Serial** (`raw_cli_executor.py`, then the permission store): `25905` hardline floor → `26006` shell hints →
  `26012` per-arg allow rules
- **Parallel**: `25904` output spill · `26007` catalog ranking · `26009` secret redaction · `26011` denial copy

### Lane 4 — Providers & caching

- **Serial** (`LLM_Calls/LLM_API_Calls.py`): `26014` cache TTL → `26015` cache key → `26022` credential borrow
- **Parallel**: `26023` models.dev catalog · `26024` auxiliary model routing

### Lane 5 — Scheduling

`26025` heartbeat → `26026` run ledger → `26027` incidents → `26028` preflight

**Owns `26004` (global emergency stop), with a caveat.** It is assigned here because the scheduler is its more
demanding reader, but it also edits `Agents/agent_service.py` and so straddles lane 1. Run it when lane 1 is
quiescent — see the cross-lane collisions below.

### Lane 6 — MCP / interop

- **Serial** (`MCP/client.py` + `MCP/local_store.py`): `25900` HTTP/SSE transport → `26032` OAuth →
  `26029` sampling & elicitation → `26013` spawn guard
- **Parallel**: `26030` resources-as-tools · `26031` outbound webhooks

### Lane 7 — Config & ops

- **Serial** (`config.py`): `26036` last-known-good → `26038` hot reload → `26040` migrations →
  `26039` unknown-key validation
- **Parallel**: `25914` Prometheus gate · `26037` faulthandler

### Lane 8 — Console surfaces

- **Serial**: `25908` /help → `25909` slash surface
- **Parallel**: `26020` @-references · `26034` terminal pane · `25906` doctor · `26035` checkpoints

### Design-gated (no code collisions, schedule freely)

`25907` cross-session memory ADR · `26033` local API server ADR · `26008` curated skill catalog

---

## Four waves

Use these instead of lanes if work is sequential rather than fanned out.

**Wave 1 — prove the rhythm.** Five tasks, zero shared files, each independently verifiable. The point is to
establish the test-and-review loop on work that cannot cascade.

`25914` · `26037` · `26009` · `26011` · `26005`

**Wave 2 — foundations that unblock others.** Seven in parallel, one per subsystem. Every task here has a
dependent waiting on it.

`25905` (unblocks 26012) · `26036` (unblocks 26038, 26040) · `26026` (unblocks 26027) ·
`25901` (unblocks 25902, informs 26002) · `25900` (unblocks 26032) · `25908` (unblocks 25909) ·
`25907` ADR (unblocks four deferred items)

**Wave 3 — dependents and independent mediums.**

`25902` · `26012` · `26038` · `26040` · `26027` · `26032` · `25909` · `26002` · `25904` · `26007` · `26014` ·
`26023` · `26025` · `26016` · `25911` · `26006` · `26030` · `26003`

**Wave 4 — the large ones.**

`25903` · `26000` · `26020` · `26034` · `26035` · `26008` · `26033` · `25906` · `26022` · `25910` · `26019` ·
`26021` · `26004` · `26039` · `26031` · `26029` · `26018` · `26017` · `26024` · `26028` · `26013` · `25912` ·
`26001` · `26010` · `26015` · `25913`

---

## Cross-lane collisions

Three places where the lane model leaks. Do not run these concurrently.

1. **`Chat/console_chat_controller.py`** — lanes 1 and 2 both touch it (`26000`/`26010`/`26004` against
   `25910`/`26017`). The lanes are safe in parallel early and collide in their late stages.
2. **`26004` global emergency stop** spans lanes 1 and 5 by design — it is one sentinel with two readers
   (agent runs and scheduled dispatch). Schedule it deliberately; do not run it beside active work in either.
3. **`26013` MCP spawn guard** is thematically a permissions task (lane 3) but edits `MCP/local_store.py`,
   which `25900` rewrites. It is placed in lane 6 after the transport work for that reason alone.

## Sequencing notes

- **Lane 1 is the critical path.** Ten tasks serialized on one file, containing `25903` and `26000`, the two
  biggest interaction changes in the batch. For the shortest wall clock, open lane 1 first and run the other
  lanes against it.
- **Do `25907` early.** It is design-only, so it costs nothing in collisions, and it blocks four deferred items
  (external memory providers, learning graph, curator, out-of-loop forks). Resolving it early keeps TASK-26041
  from stalling.
- **`26012` needs `25905` to be meaningful.** Argument-scoped allow rules on raw shell without the hardline
  floor would let a user quiet exactly the calls that most need review. The dependency is recorded on the task.
- **`26033` and `25907` both require an ADR before code.** Sweep `backlog/decisions/` for a free number at
  authoring time — ADR numbers in this repo collide routinely, and ADR-077 was renumbered twice.
