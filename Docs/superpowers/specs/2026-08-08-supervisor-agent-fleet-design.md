# Supervisor agent fleet — named sub-agents, background execution, steering, and Console supervision

**Date:** 2026-08-08
**Status:** Approved by owner (brainstorming session, section-by-section with adversarial review per section)
**Prior art:** `Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md` (the runtime this builds on), `2026-07-27-agent-programmatic-run-memory-design.md` (run log)

## 1. Purpose

Give the Console's primary agent Claude Code-style supervisor capability: it can
delegate work to **named sub-agents with user-authored instructions**, run them
**concurrently in the background**, **steer them mid-run**, and the user can
**watch and intervene** from a Console fleet panel.

Driving workflows (all four confirmed in scope): research fan-out, long agentic
tasks where delegation keeps the main context small, review/second-opinion
agents, and media/library pipelines that outlive a single chat turn.

### Owner decisions (clarifying round)

| Question | Decision |
|---|---|
| Core scope | All four: named definitions, background+parallel, multi-turn steering, live supervision UI |
| Where definitions live | DB + Settings UI (file export/import later, not now) |
| Who dispatches | Supervisor model only (user steers by talking to the supervisor; the panel watches/steers, never launches) |
| Approvals | Shared queue, per-call cards, labeled per agent — same trust model as today, multiplexed |
| UI home | Inside Console (rail panel), no new tab |
| Approach | Cross-turn architecture designed from day one, delivered in 4 phases / 6 PRs, each independently shippable |

## 2. Current state (verified against code, 2026-08-08)

- `spawn_subagent` exists (`Agents/agent_service.py` spawn closure — THE single
  spawn choke point, budget-counted, DB-lineage-tracked). The child runs its
  **entire loop inline, synchronously, mid-parent-dispatch** on the parent's
  worker thread.
- The child gets only a task string + the fixed `agents.subagent_system`
  internal prompt; inherits the parent's allow-list minus spawn/skill tools;
  result capped at `max_subagent_result_chars` (4000); `max_subagents` default
  2; depth-1 (children's `max_subagents` clamped to 0).
- Runs persist to `DB/AgentRuns_DB.py` (`agent_runs` table, parent lineage,
  `reconcile_orphaned_runs` on every DB open, `supersede_run_tree`).
- Skills also spawn children through the same closure (`allowed_tools`
  override path — must stay disjoint from this design's `agent` path).
- Console rail has a static sub-agents text line (`UI/Console_Modules/agent.py`).

## 3. Architecture overview

Two new units; the existing pure/impure split is unchanged.

**`AgentDefinition`** — a named, user-authored agent template (§4). Maps ~1:1
onto the existing frozen `AgentConfig` (model, system_prompt, allowed_tools,
budget): a definition is a stored, named way to build an `AgentConfig`.

**`FleetCoordinator`** (`Agents/fleet_coordinator.py`) — one per Console
conversation, owned by the controller (impure layer), surviving turns:

- Registry of `FleetHandle`s: `run_id`, agent definition name, task, status
  (mirrors run statuses), started-at, result text when finished.
- Per-child **inbound mailbox** (steering, phase 3) and one **outbound event
  queue** (child started / finished / needs-approval).
- **Live-children cap** checked at spawn (`[agents] max_live_subagents`,
  default 3) — the thread cap, separate from `max_subagents` (which remains
  spawns-per-turn).
- Pure state-machine logic here; thread launching stays in the service layer
  (mirrors the `agent_runtime` pure / `agent_service` impure split).

**Existing units, deltas:**

- `agent_service.py` — the spawn closure registers with the coordinator and
  launches the child via the existing `_run_one` on its own thread. The child
  loop itself is unchanged.
- `agent_models.py` — gains `AgentDefinition`, fleet event dataclasses, new
  statuses. Stays pure.
- `AgentRuns_DB.py` — gains `agent_definitions` table + columns on
  `agent_runs`. **This DB now holds durable user-authored content, not just
  telemetry** — recorded as a loud comment on the table; any future
  "clear run history" feature must not treat the file as disposable.
- Console modules — controller subscribes to fleet events; rail section grows
  into the fleet panel (§7); approval cards labeled with agent identity (§6).

**Standing invariants** (each has a dedicated test, §9):

1. **Intersection, never union.** A definition's tool list only narrows what
   the child would already have. `[tools]` gate keys always win; `risk_tags`
   flooring to `ask` is untouched (both live below this layer). A definition
   can never grant a tool the parent lacks.
2. **Depth-1 stays structural.** Children never spawn; fleet tools are pinned
   primary-only (same mechanism as `install_skill`).
3. **DB is truth, events are hot path.** `check_agents` and the panel answer
   from run statuses (DB-backed registry); the in-memory event queue is only
   the low-latency path. Restart story: children die with the process,
   `reconcile_orphaned_runs` marks them, supervisor learns next turn.
   *(Corrected 2026-08-13 with PR 3a-2 Task 7's live restart evidence:
   "learns next turn" is now only the floor. A conversation whose staged
   wake was owed at exit — a completion had landed and the durable
   `fleet_unseen` mark was set — has its owed completions delivered by the
   Console mount-claim without any user turn, exactly once per the
   `wake_delivered_at` ledger. A child killed mid-run never settled, so its
   conversation carries no mark: its swept `error` row surfaces in the
   panel and the supervisor learns of it next turn, as originally written.)*
   No new persistence machinery.
4. **Steering never cancels.** Injected instructions append messages; they
   never restart a run (cancellation-based supersede is unsound around
   durable writes — item-status lesson).
5. **Auto-wake on completion** (CORRECTED 2026-08-11 — this invariant
   previously said the opposite). A finished child wakes its supervisor so it
   can act on the result, rather than waiting for the user to return to that
   conversation and send another message.

   The original "no auto-wake" ruling rested on a false premise: that a
   supervisor turn firing with "no user present" is a problem because the user
   is watching one conversation. **This application is built around
   concurrent activity** — multiple conversations, workspaces, watchlists,
   schedules — and hopping between them. A child completing in a conversation
   the user is not currently looking at is the NORMAL case, not the edge case,
   and it is exactly the case delegation exists to serve. Deferring its result
   until the user happens to return makes background work pointless.

   This follows Claude Code's own model: a background subagent outlives the
   turn that spawned it and its completion re-invokes the parent, delivered as
   an injected notification rather than as user input.

   Two constraints the original ruling was right about, kept:
   - A completion notification must be unambiguously marked as **not user
     input**, and must never be treated as approval or consent for anything.
   - Spend is still bounded — by the containment set in §5, not by requiring a
     human keystroke.

   Consequence for the UI: completion must reach the user across
   conversations, not only inside the one that spawned the child.

## 4. Agent definitions (Phase 1)

### Data model

`AgentDefinition` (pure dataclass in `agent_models.py`; `agent_definitions`
table in AgentRuns_DB):

| Field | Rules |
|---|---|
| `name` | Unique slug (lowercase/hyphens). Reserved: `general`, `subagent`. It is a *parameter value*, never a tool name — no tool-collision validation needed. |
| `description` | One-liner the supervisor reads when choosing. Hard cap ~200 chars — it is injected into the spawn tool's schema, which fence-protocol models re-receive **every model turn** (`render_tool_protocol` dumps full parameters JSON into the system prompt), so this cap is a cost control, not polish. |
| `instructions` | Custom system-prompt body. Cap ~16k chars at the validation boundary (`input_validation.py`) — instructions ride every child model turn. |
| `tool_allowlist` | JSON array; empty = inherit everything the child would get today. Unknown names are silently dropped by intersection. |
| `model` | Optional override, **same provider/endpoint as the parent run** (the run threads one `api_endpoint`). Cross-provider children: flagged, not built. Free-text field in phase 1; a bad name errors the child run gracefully (existing machinery). |
| `enabled` | Only enabled definitions are exposed to the supervisor. |
| plus | timestamps, soft-delete flag (repo pattern). |

**Composition rule:** definition instructions **append to** the internal
`agents.subagent_system` prompt, never replace it — the base prompt carries
protocol-critical conventions (result-is-return-value, fence protocol) a
user-authored definition must not be able to break by omission.

### Spawn integration (three surgical changes)

1. `SPAWN_TOOL_SCHEMA` (static today, appended at `agent_service.py:543`)
   becomes `build_spawn_schema(definitions)`: optional `agent` parameter with
   an `enum` of enabled names (native tool-calling) **plus** one
   `name — description` prose line each in the parameter description (small
   fence-protocol models read prose better than schema). **No `agent` given =
   today's generic child, byte-identical** — phase 1 is purely additive.
2. The spawn closure resolves the definition: composed system prompt;
   allow-list = today's inherited set ∩ definition list (invariant 1). Spawn
   signature gains `agent=None` alongside the existing `allowed_tools`
   override; the two must not co-occur (asserted) — skills pass
   `allowed_tools`, never `agent`; skill path behavior unchanged.
3. `create_run` gains `agent_definition` (name) + `definition_fingerprint`
   (hash of instructions/allowlist/model at spawn time). Fingerprint, not full
   prompt snapshot — the lossless run log already records what was sent;
   the DB row just needs unambiguous audit identity for mutable definitions.

**Semantics:**

- Definitions load **once per `run_turn`** (via the service's existing
  `self.db` handle — no new wiring): the roster the model saw is exactly what
  resolves; Settings edits affect the next turn, never skew an in-flight one.
- Unknown/disabled `agent` at spawn → graceful
  `ToolResult(ok=False, error="unknown agent 'x'; available: …")` so the
  model can retry — same failure shape as the budget-exhausted refusal.

### Settings UI

**Settings ▸ Agents** category following the **About-category precedent** (a
bespoke non-TOML category: new `SettingsCategoryId` member + its own
`_render_detail_pane` branch) rendering a dedicated panel widget — the
`InternalPromptsPanel` per-item Save model is the closest shipped shape.
(Correction during PR-1 planning: `tools_settings_screen.py` is NOT usable
precedent — deprecated TASK-1346, nav-unreachable, its route resolves to the
MCP screen.) Definition list + edit form (name, description, instructions
textarea, enabled switch, tools field, model field). Explicit deviation,
stated in the UI: this edits the DB immediately (CRUD), not a config.toml
draft with save/reset. The tools field excludes **`RUNTIME_TOOL_NAMES`**
(loop machinery, not grantable) and notes that names unavailable in a given
run are ignored (intersection makes them harmless). Soft UI warning past ~20
enabled definitions (schema bloat).

### Migration

AgentRuns_DB has **no migration framework** — the established pattern
(`AgentRuns_DB.py:220-256`) is: new tables free via `CREATE TABLE IF NOT
EXISTS`; new columns via idempotent `PRAGMA table_info` + `ALTER TABLE` on
every open; then `INSERT OR IGNORE INTO schema_version (version) VALUES (5)`.
This design adds no framework — it follows that pattern exactly (also for
`resumed_from_run_id` in phase 3).

### Deliberately deferred from phase 1

- No seeded starter definitions (the Default-Assistant seeding incident —
  FTS5 triggers predating the seed INSERT — made migration-time seeding a
  known trap). Starter library lands in phase 4 as plain CRUD inserts.
- No per-definition budgets (arrives with phase-3 containment, where
  per-definition `max_wall_seconds` becomes meaningful).
- No file export/import.

## 5. Fleet execution model (Phases 2–3)

Design principle: **build the cross-turn architecture in phase 2, restrict it
to turn-scoped use until phase 3 lifts the restriction.** No throwaway
concurrency model.

### Phase 2 — concurrency inside the turn

- The spawn closure changes from "run child inline, return result" to
  "register with coordinator, launch child on its own **daemon thread** via
  the existing `_run_one`, return a handle id immediately."
- Two new runtime tools (both **primary-only**, pinned like `install_skill`):
  - `wait_agents(ids?)` — block until named (or all) children finish, return
    results. Dispatched **in-loop** like `spawn` (naturally outside the
    `max_tool_call_seconds` daemon wrapper); bounded by the parent's remaining
    wall-clock; **polls `should_cancel`** — user cancellation propagates to
    cooperative-cancel of all children in turn-scoped mode.
    **Result sizing:** per-child results are capped at
    `max_subagent_result_chars` (4000) as today, but the combined tool result
    is ALSO bounded by `max_tool_result_chars` (16000) at the history-append
    seam — 5 children × 4000 chars would truncate mid-results. `wait_agents`
    therefore allocates the history budget **evenly across returned children**
    (each entry truncated with a notice); the supervisor re-fetches one
    child's full capped result by calling `wait_agents([id])` for it alone.
- The Console bridge raises `max_subagents` alongside the live-children cap
  (`CONSOLE_RUN_BUDGET` currently inherits the default of **2 spawns/turn** —
  left alone, phase 2's parallelism caps out at 2 before the fleet cap ever
  binds). Sized together with `[agents] max_live_subagents`, config-visible.
  - `check_agents()` — non-blocking status snapshot from the DB-backed
    registry (invariant 3).
- The spawn tool description teaches the spawn→`wait_agents` pattern; results
  arriving after the model's final text are wasted.
- **End-of-turn safety net:** if the loop ends with children running, the
  service waits (bounded by parent remaining wall-clock), then
  cooperative-cancels stragglers, then **abandons** a wedged thread after a
  5s join timeout (same precedent as `_call_with_timeout`'s daemon
  abandonment).

### Phase 3 — crossing the turn boundary

1. End-of-turn no longer waits; children may outlive the turn. Nothing new
   persists (runs already in DB; reconcile already handles process death).
2. Completion **wakes the supervisor** (corrected 2026-08-11; see §3
   invariant 5). A finished child injects a completion notice and re-invokes
   its supervisor so it can act — it does not sit in a queue waiting for the
   user to send another message in that conversation. The model fetches full
   results via `wait_agents`, which returns immediately for finished children;
   results also live in the run row's `result` column, so they survive
   restart. A Console chip surfaces the same event to the user, reachable
   from any conversation.

   The injected notice must be unambiguously marked **not user input** and
   must never be read as approval or consent — the one thing the original
   no-auto-wake ruling was right to worry about.
3. `send_to_agent(id, message)` + user steering from the panel (§6).
4. **Wiring-lifetime audit — the hard part of 3a.** A live child holds the
   prior turn's object graph: the approval-routing callable (bound "for THIS
   run"), the per-run `ToolCatalogRegistry` + provider instances, the
   run-tree-bound run-log writer, `on_step`, diff sink, cost hooks. Python
   keeps them alive via the thread's references, but *alive* ≠ *valid*: each
   seam is audited for behavior after its turn ends. The run-log writer and
   lineage are correct by construction (the child belongs to its spawn turn's
   tree); approval accounting must keep accepting rounds from a past turn's
   run; anything found to be turn-scoped by assumption gets rebound or
   documented as intentionally frozen.

### Containment (replaces `clamp_child_budget`'s parent-remainder clamp)

**Timing:** phase 2 keeps `clamp_child_budget` byte-identical — turn-scoped
children must die by end of turn anyway, so the parent-remainder clamp is
still the correct bound. Phase 3a swaps in the set below.

A background child deliberately outlives its parent — the old invariant
("child never outlives parent") is replaced, not just deleted:

- Per-child `max_wall_seconds` from the definition (phase 4) or config default.
- Live-children cap (`[agents] max_live_subagents`, default 3).
- Per-child `max_total_tokens` passed through as today.

Bounded in time, count, and spend — just no longer by the parent's lifetime.

### Concurrency corrections (all verified against code, all phase-2 items)

| Finding | Fix |
|---|---|
| `set_status` is an unconditional UPDATE (`AgentRuns_DB.py:498-503`); an abandoned child finishing late overwrites the coordinator's `cancelled` with `done` | Terminal-transition guard: `WHERE status NOT IN (…terminal…)` — first writer wins. |
| `on_step` is called as `(step, agent_kind)` (`agent_service.py:1303`) — N children's steps all arrive labeled `"subagent"`, unattributable | Signature carries `run_id` + definition name; bridge routes by it (small compat shim for the existing step display). |
| Shared `ToolCatalogRegistry` name→id/owner caches race under concurrent rebuild — the code already warns (`tool_catalog.py:843-846`) | Lock around cache rebuild/resolution. Providers stay shared (they must); no per-child registries. |
| `supersede_run_tree` supersedes the whole tree — in phase 3, the first user message while background children run would kill the live fleet | Supersede applies to the **primary** run; the sweep skips non-terminal background children (rows stay parented for lineage). Inverts a deliberate invariant → dedicated tests. |
| **Both** permission gates hold nesting-shaped per-turn state: `MCPToolProvider._stamped_decisions` (REPLACED every turn, `stamp_scope` snapshot/restore is LIFO-only) **and** `BuiltinToolGate` (`stamp_scope` + `begin_turn` clears stamps on a shared instance mid-siblings) | Verdict state keyed `(run_id, tool)` in both gates; `begin_turn` becomes per-run. Replaces the C1 `review_state_scope` seam for concurrent children. |
| Tool providers were written under one-run-at-a-time dispatch | Phase-2 thread-safety audit (MCP control-plane client, local tools, gated builtins). Unaudited provider ⇒ per-provider execution lock on `invoke` (throttle, not break). **MCP starts locked until proven otherwise.** |
| Run-log writer under concurrent children | **Verified safe** — already lock-protected (`run_log.py:481,746`). No change. |

Provider-side reality, stated honestly: N parallel children = N concurrent LLM
streams; the practical ceiling is provider rate limits and spend — which is
exactly why the cap is config, not architecture. Nothing in the coordinator,
event queue, DB layer, or coalescer assumes small N (the long-term vision
shows fleets of dozens).

## 6. Steering and approvals

### Steering — two paths, one mechanism

- Supervisor: `send_to_agent(id, message)` tool. User: input in the fleet
  panel drill-in. Both append to the same per-child mailbox.
- Entries become **user-role messages drained between the child's model
  turns** (clean insertion point in `run_agent_loop`, before each model call).
  Never cancel/restart (invariant 4). Drained only at a **protocol-coherent
  boundary** — after every pending tool result for the previous assistant
  message has been appended — so native tool-call pairing (`tool_calls` ↔
  `role:"tool"` ids) is never split by an injected message.
- **Source labels:** injected messages are prefixed
  `[Steering from supervisor]` / `[Steering from user]` in both the message
  and the run-log record — the child must not mistake supervisor text for the
  human's voice, nor the audit trail.
- **Latency honesty:** a child in a long tool call sees steering late; the
  panel shows "queued" until consumed.

### Steering a finished agent (phase 3)

Claude Code's SendMessage continues *completed* agents; "course-correct
instead of respawn" includes "now also check X" after the reviewer finished.

- The coordinator retains finished children's message transcripts **in
  memory** (cap: last 5 finished per conversation, ≤200k chars each;
  oldest evicted first).
- `send_to_agent` to a finished child starts a **new run seeded with the
  retained transcript** + the steering message, linked via
  `resumed_from_run_id` (idempotent-ALTER column).
- After app restart the transcript is gone → clear error suggesting a fresh
  spawn. Cross-restart resurrection is explicitly out of scope (rebuilding
  history from the run log is forensics machinery masquerading as a feature).

### Approvals — extend the existing gate, no second gate

- Cards carry agent identity (definition name + task snippet).
- **A blocked child pauses only itself** — structurally true already: the
  approval wait happens inside `_call_with_timeout`'s per-call daemon thread
  (`console_chat_controller.py:125-132`), so on a child's own thread it
  isolates for free. Siblings and the supervisor continue.
- One visible queue, arrival-order across the fleet.
- Approval accounting is already **round-keyed** (`_pending_approvals`,
  TASK-1050) — audit that round keys are globally unique across concurrent
  runs; if per-run sequential, key by `(run_id, round)`.
- Timeout unchanged; the deliberate invariant
  `approval_timeout < max_tool_call_seconds` (documented at
  `console_chat_controller.py:121-132`) is preserved untouched.
- **Zombie-card revocation (safety item):** cancelling or abandoning a child
  revokes its pending approval rounds — cards auto-dismiss as denied, at the
  coordinator's cancel path. Without this, a user can approve a card for an
  already-abandoned child and the tool executes for real (file written,
  message sent) on the abandoned daemon thread while the run reads
  `cancelled`.
- **Session-scope grants stay session-wide** (TASK-1861 scopes are keyed by
  tool name), but the card says it plainly: "Allow for this session (applies
  to all agents)". Honest labeling over per-agent grant machinery.

## 7. Console fleet UI — an Inspector section

### Long-term direction (owner-supplied vision, two screenshots 2026-08-08)

The rail evolves toward a Claude Code-desktop-style **Inspector**: stacked
sections with a shared grammar — section header + chevron, icon + label +
right-aligned status rows, "View all" tail, glyph status dots
(`console_glyphs.py` carries this idiom in Textual). Long-term sections and
their existing substrate:

| Inspector section | Substrate today |
|---|---|
| **Agents** (this program) | Coordinator + run-log program + this design |
| **Changes** (+N −M) | `change_snapshots` table (`record_change_snapshot`, per-conversation queries) + TASK-1366 `diff_sink` UI channel — data already flows; needs aggregation + a section |
| **Scoped RAG data / Sources** | Console auto-retrieve (task-406) knows what it retrieved per turn |
| **Workspace / git / PR, multi-root** | `console_workspace_context.py` for workspace identity; git/PR status needs a new collector (only section with no substrate) |

**Scope discipline:** this program ships the reusable **section component**
and the **Agents section** on it. Changes/RAG/Git-PR/multi-root are filed as
backlog follow-ups referencing the component — recorded here so nothing built
now fights the target, and nothing long-term sneaks into this program's ACs.

### Agents section — three states

1. **Summary line** (collapsed; what most sessions see): glyph cluster +
   `N working` with `M done` right-aligned — the grammar from the owner's
   second screenshot. Cheapest state for the coalescer to feed.
2. **Expanded rows**: one per child, **two-line** (line 1: glyph + name +
   elapsed; line 2: dimmed last-step summary, truncated) — the rail clips
   single dense lines at default width (task-226 precedent). Scrollable /
   virtualized past a screenful; "View all" opens full run history. Every row
   width-bounded.
3. **Drill-in**: child transcript from the coalesced step buffer; Cancel
   (cooperative + card revocation); phase 3 adds the steering input +
   mailbox "queued" state. Wiring the run-log viewer with a live-follow mode
   is an **audit, not a dependency** — it lands only if genuinely
   incremental; the step buffer is the baseline.

### Mechanics

- **Coalescer:** fleet events batch ~250ms on the UI side; row widgets
  refresh individually (`recompose=False`); the panel never rebuilds per step
  (task-3010 lesson).
- **Notifications:** completion bumps the badge + low-key chip; no modal, no
  focus steal. **Cross-conversation by requirement, not as a follow-up**
  (corrected 2026-08-11 alongside §3 invariant 5): the user hops between
  conversations, workspaces, watchlists and schedules, so a child finishing
  in a conversation that is not on screen is the normal case. A completion
  the user can only discover by returning to the right conversation is a
  completion they will miss. The indicator must be reachable from wherever
  they are; the fleet panel's own rows stay scoped to the conversation they
  belong to.

  **Where the wake itself runs — the honest architectural limit as built
  (recorded 2026-08-13, PR 3a-2).** The INDICATOR is fully cross-screen
  (app-wide toast + durable unseen-completion mark + session glyphs), but
  the wake TURN requires a live Console controller: bridge and controller
  are per-screen today, so the wake fires immediately whenever a Console
  screen is mounted — whichever session or screen the user is watching —
  and with no Console mounted it is staged durably (the mark plus the
  per-run `agent_runs.wake_delivered_at` ledger) and claimed synchronously
  at the next Console mount. A fully headless supervisor — one that ACTS
  with no Console mounted at all — requires moving bridge/controller
  ownership above the screen and is a filed follow-up (task-15860).
  Everything PR 3a-2 built is substrate that version needs; the only delta
  is where the wake runs.

  **SUPERSEDED 2026-08-17 — the limit above no longer holds (task-15860,
  eight merged landings).** The Console runtime (agent bridge +
  `ConsoleChatController` + the message store) is owned by the APP, not by
  `ChatScreen`; a screen attaches as a view and detaches at unmount, and
  `ConsoleFleetWakeCoordinator._attempt` now refuses only a DISPOSED
  controller (app exit), never a merely-ended visit. So a background
  sub-agent that finishes with no Console mounted delivers a full
  supervisor turn there and then, and a completion owed across a restart
  is delivered at the next launch with Console never opened (mark-gated,
  behind the same `[agents] autowake_enabled`). The app-owned store is the
  continuity owner, so that turn is in the transcript when the user
  returns. Driven live 2026-08-17 against a real provider (see
  `Docs/User_Guide/console/agent-runs-and-tools.md`'s stamp and
  `Docs/superpowers/plans/2026-08-14-headless-wake-closeout-report.md`).
  Two honest residues, both recorded rather than hidden: a process killed
  in the window between a wake turn's acceptance and its ledger stamp
  costs exactly one re-announce at the next launch (never a lost result),
  and a headless approval round's card currently mounts empty until the
  user clicks that session's tab (task-17500).

  *(Corrected 2026-08-14, wake-integrity arc — tasks 15970/15971, per the
  coordinator's design ruling; rationale re-anchored the same day, see the
  superseding note below.)* A Console screen that is mounted but not the
  one being LOOKED AT — covered by a pushed screen (command palette, nav
  overflow menu, picker, dialog), or displayed with a different session
  tab active — delivers its wake anyway. **That off-view delivery is the
  intended behavior**, and the reason is the invariant, not the plumbing:
  a supervisor exists to act on its children's results the moment they
  land (§3 invariant 5). Staging is the fallback for when there is no
  controller to act with, never a politeness about visibility. What the
  user is owed instead of staging is *knowledge*: the settle toast fires
  at settle from any screen, and a wake turn that completes while its
  conversation is not the displayed-and-active one **leaves the
  `FLEET_UNSEEN` mark set** (delivery-commit visibility probe,
  `wake_conversation_in_view`), so the `◈` badge points at the delivered
  result until the user views it — view-clear applies normally then,
  since a delivered wake has nothing pending. "Viewing" means DISPLAYED:
  a mounted-but-undisplayed Console's sync tick never view-clears the
  mark.

  *(Superseding correction, 2026-08-14 — task-16300.)* The ruling above
  stands; the rationale first recorded for it does not. That text argued
  off-screen delivery was intended because *"'whenever a Console screen
  is mounted' includes a Console screen the user has NAVIGATED AWAY
  FROM"* — a Chat screen staying resident in the screen stack after
  navigation. **That was never a nav model; it was a bug.** Textual's
  `App.switch_screen` pops only the top of the screen stack, so a
  navigation issued while any pushed screen sat above Chat replaced the
  MODAL and left Chat resident — violating the invariant `app.py`'s
  `_create_navigation_screen` documents (screens die on navigation;
  `ScreenStateStore` carries continuity), and producing exactly the two
  failures 15970/15971 were filed for. Navigation now reduces the stack
  to its content screen before switching, so **a hidden-but-resident
  Chat screen is no longer a normal state**: leaving Console unmounts it
  and shuts its controller down, on every path. Nothing about the ruling
  changes — off-view delivery and the `◈` guarantee are unaffected,
  because the off-view cases that matter (a modal covering Console, a
  non-active session tab) never depended on residency. What changes is
  the boundary of the staging path: durable staging (mark +
  `wake_delivered_at` ledger + mount-claim + session-open trigger) is the
  path whenever no Console screen is mounted at all — restart, first
  boot, **and navigating away from Console**, which is genuinely
  unmounted again rather than secretly still running.
- **Cost:** per-child token spend rolls into the existing Console cost
  ticker, attributed per agent in expanded rows; fleet aggregate visible on
  the summary line's expansion.
- **Thread discipline:** all fleet events cross to the UI via the same
  `call_from_thread`/message-posting seam `on_step` uses; coordinator threads
  never touch widgets.
- **Tests assert rendered geometry**, not DOM presence (unbounded-width
  Statics render invisible to headless queries — Library-UAT lesson).

## 8. Error handling

- Child error/stuck → honest status in panel; supervisor sees `ok=False` +
  status text from `wait_agents` and decides retry/respawn itself. No retry
  machinery in v1.
- Provider rate limits under parallelism → per-child run errors, graceful.
- Process death → `reconcile_orphaned_runs` (runs on every DB open) + panel
  shows reconciled statuses; supervisor learns next turn *(corrected
  2026-08-13, PR 3a-2 Task 7 live evidence: when the conversation was
  already marked by a pre-exit completion, the mount-claim delivers the
  owed wake without a user turn — "next turn" remains true only for
  mid-run orphans, whose conversations were never marked)*; retained
  transcripts lost with clear `send_to_agent` error.
- Guards: terminal-status first-writer-wins; cancel revokes pending cards;
  supersede spares live background children.
- **Stop semantics:** phase 2 — Stop cancels supervisor + children
  (turn-scoped, unchanged mental model). Phase 3 — Stop cancels the
  supervisor's turn only; the panel offers explicit "Cancel all agents".
- Conversation deletion, or a temporary ("not saved") session closing,
  cancels its fleet (cooperative cancel + card revocation). Ephemeral runs
  already flow through the registry's `_ephemeral` gate at `invoke_by_name`.
- Phase 3b audits the cost ticker's assumption that spend accrues only
  during a turn — background children accrue between turns, delivered
  through the coalescer.

## 9. Testing

Repo norms apply: real in-memory SQLite, targeted test files (owner ruling:
no routine full-suite sweeps), Hypothesis where shapes warrant, live
verification per `backlog/docs/lessons-live-verification.md`.

- **Concurrency:** threaded children writing steps/status concurrently;
  terminal-guard race; registry-cache race; per-run gate-scoping probe tests
  (C1 style — probe the clobber, then prove the fix) for BOTH gates.
- **Invariants (§3):** intersection-never-grants; depth-1 with definitions;
  supersede-spares-fleet; zombie-card revocation; steering-never-cancels.
- **Phase 1:** byte-identical no-`agent` spawn; load-once-per-turn; unknown
  agent error shape; migration idempotency (open old DB twice).
- **UI:** rendered-geometry assertions for summary line and rows; coalescer
  batching; per-conversation panel swap.
- **Live (before each phase merges):** tmux TUI against a real provider
  (repo-root API keys are for agent use): one genuine research fan-out, one
  steering exercise, one cancellation with a pending approval card.

## 10. Phase → PR map

| PR | Contents |
|---|---|
| **1 — Definitions** | Models + DB (idempotent ALTERs, version row 5) + spawn `agent` param + Settings ▸ Agents editor + user-guide page |
| **2a — Concurrency runtime** | FleetCoordinator, threaded children, `wait_agents`/`check_agents`, both gates per-run scoped, registry lock, `on_step` run_id, `set_status` guard, approval-round keying, MCP provider locked pending audit, card revocation |
| **2b — Panel v1** | Section component + Agents section (summary/rows/drill-in read-only), coalescer, cost rollup |
| **3a — Cross-turn runtime** | (corrected 2026-08-13, following §3 invariant 5's / §7's 2026-08-11 corrections; shipped as two PRs) **3a-1:** end-of-turn no longer waits, supersede boundary change (merged, PR #1557). **3a-2:** auto-wake on completion + cross-conversation notification — the completion **wakes the supervisor**; this row's original "completion delivery next turn" was the pre-correction ruling. Mailboxes + `send_to_agent` + finished-agent continuation (`resumed_from_run_id`) move to 3b with the steering UI |
| **3b — Steering UI** | Steering input, mailboxes + `send_to_agent`, finished-agent continuation (`resumed_from_run_id`) (both moved from 3a, 2026-08-13), mailbox "queued" state, notification chip, Stop-semantics change + "Cancel all agents" |
| **4 — Polish** | Starter library (plain CRUD: researcher, critic, ingest-runner), per-definition `max_wall_seconds`, config knobs, docs pass, file follow-ups (inspector sections) |

Backlog: one parent task + one per PR, IDs assigned against origin/dev with
headroom (collision lesson). UI-changing PRs update the matching
`Docs/User_Guide/` page per CLAUDE.md.

### Config surface

```toml
[agents]
max_live_subagents = 3        # fleet thread cap (phase 2)
autowake_enabled = true       # completion wakes the supervisor (3a-2 kill switch; default ON)
# retained finished transcripts: count + byte caps (phase 3)
```

Definitions themselves live in the DB, not config.

## 11. Out of scope / follow-ups (filed, not built)

- ~~Auto-wake / auto-resume of the supervisor on child completion (config-gated
  follow-up at most).~~ (corrected 2026-08-13 — this bullet contradicted §3
  invariant 5's 2026-08-11 correction, which makes auto-wake REQUIRED. Built in
  PR 3a-2: `[agents] autowake_enabled` defaults ON and is a kill switch, not a
  scope reduction. What genuinely remains a follow-up is the **headless** wake —
  the supervisor acting while no Console screen is mounted — filed as
  task-15860.) **CLOSED 2026-08-17**: task-15860 shipped in eight
  landings. Nothing about the wake remains a follow-up; the app owns the
  Console runtime, the wake fires with no Console mounted and at launch
  with Console never opened, and §7's "honest architectural limit as
  built" paragraph carries its superseding note. What is still open is a
  BUG found by the close-out live pass, not a scope gap: task-17500 (a
  headless approval card mounts empty until the session tab is clicked).
- Cross-provider child models; provider-aware model validation in the editor.
- Cross-restart resurrection of finished agents.
- File export/import of definitions.
- Inspector sections beyond Agents (Changes, Sources/RAG, Workspace/Git/PR,
  multi-root) — filed referencing the section component.
- ~~Cross-conversation completion indicator.~~ (corrected 2026-08-13 — required
  by §7 as corrected 2026-08-11: "cross-conversation by requirement, not as a
  follow-up". Built in PR 3a-2: app-wide completion toast + durable
  `FLEET_UNSEEN` mark + `◈` session marker + mount-time claim.)
- Per-agent session approval grants (kept session-wide, labeled honestly).
