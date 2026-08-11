# Supervisor Fleet PR 3a-1 — Cross-Turn Child Lifetime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a sub-agent keep working after the turn that spawned it ends — correctly, with every turn-scoped resource it depends on either given a longer life or deliberately frozen.

**Architecture:** Phase 2 made children concurrent but strictly turn-scoped: `_settle_fleet` waits, cancels, then abandons every child before `run_turn` returns. This PR removes that hard boundary and repairs the four seams that boundary was hiding. It does **not** add steering or completion delivery — those are PR 3a-2.

**Tech Stack:** Python ≥3.11 (`threading`, `asyncio`), Textual 8.x, SQLite WAL, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md` §5 "Phase 3" + the Containment subsection. Read both before Task 1.

## Why this is split from 3a-2

The spec's Phase 3 bundles lifetime, completion delivery, steering, and finished-agent continuation. A survey of the actual seams (below) shows lifetime alone is a full PR: the hardest dependency is **actively destroyed** at end of turn, not merely stale. Shipping "children survive" separately from "children can be steered" keeps each independently reviewable, and 3a-1 is what 3a-2 stands on.

## Global Constraints

- Worktree `.worktrees/fleet-pr3a`, branch `feat/fleet-cross-turn`, cut from dev. NEVER git outside it.
- **NEVER `git stash`** — the stack is shared across 100+ worktrees; a pop can restore another session's uncommitted work here and drop it from theirs. Compare with `git show HEAD:<path>`, `git diff`, or /tmp copies.
- Never `git checkout` a file to undo an edit. Edit-based restores only, with `git diff` proving byte-identical restoration.
- pytest is the ONLY python entry point. A bare `python -c` importing `tldw_chatbook.config` triggers the app's config rewrite and has written to the user's LIVE config this session. Never touch `~/.config/tldw_cli` or `~/.local/share/tldw_cli`.
- Never hand-edit `css/tldw_cli_modular.tcss` (generated).
- **Turn-scoped behaviour must stay byte-identical when no child outlives its turn.** That is the overwhelmingly common case, and the existing fleet suites are its guard.
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

### Verified seam map — cite these, do not re-derive

| Seam | Location | Lifetime today |
|---|---|---|
| `_settle_fleet` (wait → cancel → 5s join → abandon) | `agent_service.py:868-947`, called `:2445-2459` | Turn |
| **asyncio loop + driver thread** | built `console_agent_bridge.py:2009-2016`, **torn down** `:2265-2286` | Turn — *destroyed*, see below |
| `RunLogWriter` | built `agent_service.py:2394-2400`, `bind()` latches (`run_log.py:502-511`), **`close()`** `:2479` | Run tree |
| `ToolCatalogRegistry` + providers | `console_agent_bridge.py:1367-1394` per `run_reply` | Turn |
| `on_step` / diff sink / cost signals | `console_agent_bridge.py:2031`, `:1660`, controller `:8274` | Turn |
| `review_tool_calls` hook | `console_chat_controller.py:9106` | Turn |
| `revoke_approval_rounds_for_run` + round registry | controller `__init__` `:1226`, `:1432`; method `:4052-4145`, **run-keyed** | Controller (survives) |
| `AgentService` + `_fleet*` | `console_agent_bridge.py:2173`; state `agent_service.py:506-509`, reset `:2422-2423` | Turn |
| `clamp_child_budget` | `agent_models.py:358-396`, sole call `agent_service.py:1285` | — |
| `supersede_run_tree` | `AgentRuns_DB.py:943-962`, sole prod call `agent_service.py:2389` | — |
| `reconcile_orphaned_runs` | `AgentRuns_DB.py:678-742`, called from `__init__` `:57`, once per process | — |
| `FleetCoordinator` | `fleet_coordinator.py:66-256` — already reusable, no reset, never pruned | Reusable |

**Already anticipating this PR** (do not fight these comments — they are the design intent): `agent_service.py:891-894` (settle must not reach another turn's children), `:1154-1159` (`my_handle_ids` exists precisely because the coordinator is not run-scoped), `:479-489` (an injected coordinator is caller-owned).

---

### Task 1: Decide and pin the model-call lifeline

**This task is a decision plus its enabling change. Do it first; everything else depends on the answer.**

A cross-turn child calls the model through `_StreamingModelAdapter`, which submits with `run_coroutine_threadsafe` onto the per-run loop built at `console_agent_bridge.py:2009-2016`. That loop is stopped, joined and **closed** in `run_reply`'s `finally` (`:2265-2286`), whose comment states the current invariant plainly: *"By this point `run_turn` has already settled every fleet child … so nothing should still be submitting."* Once children outlive the turn, that sentence is false and the loop is gone.

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Test: `Tests/Chat/test_console_agent_bridge.py`

- [ ] **Step 1: Establish the constraint before choosing**

Read PR #629's property that the current design protects: one loop per run means one owned `httpx` client per run (`ConsoleProviderGateway._active_http_client` caches by running loop). Write down, in your report, what breaks if two runs share a loop and what breaks if a child gets its own.

- [ ] **Step 2: Choose between these three, and justify it**

**The decision is made: a fleet child owns its own model-call lifeline from
birth.** It never borrows the turn's loop, so there is nothing to transfer at
settle time and nothing that dies when the turn ends. This is Claude Code's
own model — a background subagent is an independent runtime, which is why its
lifetime is not a problem there.

Your job is not to re-decide it; it is to implement it and measure the one
thing it costs. `ConsoleProviderGateway._active_http_client` caches by running
loop, so per-child loops mean per-child HTTP clients where three children
currently share one. Measure that (client construction cost, fd count, and
whether the gateway's own locking assumes a single client) and report numbers,
not adjectives. If the measurement says per-child clients are genuinely
expensive, say so with the figures and propose a shared-client-across-loops
variant — do not silently fall back to the turn-scoped loop.

Do NOT choose option (C)-style "refuse to survive if a call is in flight":
survival must be predictable from the user's side.

- [ ] **Step 3: Write the failing test first**

A child that is still running when `run_reply` returns must be able to complete a model call afterwards. Assert on the child's persisted run row reaching a terminal status with real content, not on internal loop state.

- [ ] **Step 4: Implement, then prove the common case is untouched**

Run `pytest Tests/Chat/test_console_agent_bridge.py Tests/Agents/ -q` — READ the counts. The turn-scoped suites are the guard that nothing changed when no child survives.

- [ ] **Step 5: Commit**

---

### Task 2: Make `_settle_fleet` conditional

**Files:** `tldw_chatbook/Agents/agent_service.py`; test `Tests/Agents/test_fleet_runtime.py`

Today `_settle_fleet` unconditionally waits, cancels and abandons (`:911-947`). PR 3a-1 keeps that behaviour for a **turn-scoped** run and skips it for children explicitly allowed to outlive the turn.

- [ ] **Step 1: Decide the opt-in shape.** A per-definition flag, a config knob, or "every child survives by default"? The spec's §7 UI shows a fleet the user watches across turns, which argues for survival being normal — but the byte-identical constraint argues for an explicit switch. Choose, and say why in the report.
- [ ] **Step 2: Failing tests** — (a) a surviving child is NOT cancelled when the turn returns and its run row is not forced to `cancelled`; (b) a turn-scoped child still settles exactly as today (this is the regression guard); (c) the manifest-ordering constraint at `:2445-2459` still holds for whatever the turn does settle.
- [ ] **Step 3: Implement.** Keep `mine = list(self._fleet_cancels)` scoping — the comment at `:891-894` explains why settling must never reach another turn's children, and that becomes load-bearing here.
- [ ] **Step 4: Gate** `pytest Tests/Agents/ -q`, READ counts. **Step 5: Commit.**

---

### Task 3: Run-log writer lifetime

**Files:** `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/Agents/run_log.py`; test `Tests/Agents/test_run_log*.py`

**CORRECTED after Task 2's review — read this before touching anything.** The
obvious diagnosis is wrong, and acting on it ships a non-fix.

It is NOT (only) that a survivor appends to a *closed* writer. `on_record`
reads `self.run_log_writer` **at call time** (`agent_service.py:2368`), and
`run_turn` **replaces that attribute** with a fresh writer bound to the new
primary (`:2564-2569`, `:1165`). Probed with a recording-writer double: turn
1's survivor wrote **zero** records to turn 1's writer, and its `model` record
landed on turn 2's writer, bound to **turn 2's primary run id**.

Two consequences, both worse than a dropped append:
- Turn 1's child has an **empty** "Full run log".
- Turn 2's run tree contains a **foreign run's records**, reachable via
  `run_log_slice`/`search_run_log` scoped to that tree — the exact inverse of
  the property `test_run_log_sandbox_isolation` and
  `test_run_log_workspace_isolation` exist to defend.

So deferring `close()` fixes nothing: the attribute swap is untouched by
deferral, and the misfiling happens between turns, not after them. The writer
a child records through must be resolved **per run**, not read off the service
at call time.

Also still true: `bind()` latches permanently (`run_log.py:502-511`) and a
writer is scoped to one run tree by design (`:2378-2389`) — do not "fix" this
by reusing one writer across trees.

- [ ] **Step 1: Failing test** — reproduce the MISFILING first, not the closed-writer story: a survivor's records after its turn ends must not appear in the next turn's tree, and must land in its own tree or be dropped deliberately and observably. A silent drop is not acceptable; silent misfiling into another tree is worse.
- [ ] **Step 2: Implement.** Bind the writer a child records through to the child's own run tree rather than to whatever `self.run_log_writer` currently points at. Weigh in the report: pass the writer down at spawn; defer the tree's `close()` until its last child finishes (necessary but NOT sufficient on its own — say why in your report to prove you read this); or give the survivor its own writer. Whatever you choose, add a test that the isolation properties above still hold with a survivor in flight.
- [ ] **Step 3: Gate + Commit.**

---

### Task 4: Stop supersede from killing live children

**Files:** `tldw_chatbook/DB/AgentRuns_DB.py`; test `Tests/DB/test_agent_runs_db.py`

`supersede_run_tree` (`:943-962`) flips `id = ? OR parent_run_id = ?` to `superseded` with **no liveness guard**. Today that is coherent because children cannot outlive the turn. After Task 2, the first user message sent while background children run would mark the live fleet superseded — the feature destroyed by an existing correctness mechanism.

- [ ] **Step 1: Failing tests** — (a) superseding a primary run leaves its still-`running` children untouched, while terminal children are superseded as before; (b) **the lost-result half**: a live child superseded mid-flight must still persist its real terminal status and result when it finishes — assert the result is readable afterwards, not merely that the row is not `superseded`.
- [ ] **Step 2: Implement** — narrow the UPDATE so a non-terminal child is skipped. Keep lineage intact (rows stay parented); this changes status semantics only.
- [ ] **Step 3: Gate** `pytest Tests/DB/test_agent_runs_db.py Tests/Agents/ -q`. **Step 4: Commit.**

---

### Task 5: Replace the parent-remainder clamp with real containment

**Files:** `tldw_chatbook/Agents/agent_models.py`, `tldw_chatbook/Agents/agent_service.py`; tests `Tests/Agents/test_agent_models.py`, `test_fleet_runtime.py`

`clamp_child_budget` (`:358-396`) exists to guarantee "a child can never outlive its parent" — the exact invariant this PR removes on purpose. Deleting it without a replacement leaves a surviving child unbounded.

Replacement (spec §5 Containment): per-child `max_wall_seconds` from the definition or a config default; the existing live-children cap; `max_total_tokens` passed through as today. Bounded in **time, count and spend** — just not by the parent's lifetime.

- [ ] **Step 1: Failing tests** — a surviving child is still bounded by its own wall-clock and token ceiling; `max_subagents=0` (depth-1) is preserved; a turn-scoped child's budget is unchanged.
- [ ] **Step 2: Implement.** Keep the function for the turn-scoped path if that is cleanest; do not silently widen the depth-1 guarantee.
- [ ] **Step 3: Gate + Commit.**

---

### Task 6: The turn-scoped closure audit

**Files:** `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_chat_controller.py`; tests as needed

For each seam below, a surviving child must either keep a valid one, get a replacement, or have its output deliberately and observably dropped. **Silent loss is the failure mode to avoid** — a child whose steps, diffs and token spend vanish looks like it worked.

**The failure shape to hunt, named by Task 3** (which found it the hard way):
*per-turn state read off `self` at call time silently misfiles work that
outlives the turn.* The run-log writer had exactly this shape — `on_record`
read `self.run_log_writer` at call time while `run_turn` replaced the
attribute, so a survivor's records landed in the next turn's tree. Nothing
raised; the records simply went somewhere wrong. Task 3 fixed it by resolving
the writer per RUN and capturing it on the parent's thread **at spawn time**,
so a child not scheduled until the next turn still files into its own tree.

Sweep for the same shape systematically, not opportunistically. Every
`self.<per-turn attr>` read from a code path a child can reach is a candidate:
start with `_turn_definitions`, `_fleet`, `_fleet_cancels`, `_fleet_threads`,
`registry`. For each, answer: is it read at call time or captured at spawn?
If read at call time, what does a survivor get after `run_turn` reassigns it —
and does anything raise, or does it silently do the wrong thing? Silence is
the dangerous answer.

- [ ] **Step 1: Audit and report a table** (seam → what a surviving child does today → decision → evidence):
  - `on_step` (`console_agent_bridge.py:2031`) — closes over turn-local `live_steps`/`subagents`/`pending_diffs`
  - diff sink (`:1660`, wired `:1711`) — nothing drains it after the turn
  - cost signals (`console_chat_controller.py:8274`, `_attach_stream_usage` `:8334-8419`) — bound to one `assistant_message_id`
  - `review_tool_calls` hook (`:9106`) and the gate state it closes over
  - `ToolCatalogRegistry` + providers (`:1367-1394`)
  - `_turn_definitions` (`agent_service.py:2411-2414`) — frozen per turn *by design*; confirm freezing is still right for a surviving child and say so explicitly
  - **`ConsoleProviderGateway` / the owned HTTP client** (added after Task 1's
    review — this list is an enumeration, so an omitted seam is a missed
    seam). `gateway.aclose()`, called from `ChatScreen.on_unmount`, pops the
    calling loop's client and schedules a stale-close for **every other
    cached loop's** client, skipping only *closed* loops — and a live child's
    loop is not closed, so its pool gets closed on its own loop mid-request.
    Note this is **not** introduced by the fleet: it already reaches the
    PRIMARY today, because `run_reply` is dispatched via `asyncio.to_thread`,
    which survives Task cancellation (the controller's own comment at
    `console_chat_controller.py:1322-1324` says so), so `on_unmount` can call
    `aclose()` while a turn loop is still live. Per-child lifelines enlarge
    the population from one loop to one-per-live-child. It becomes
    user-visible from Task 2 onward, when children genuinely survive.
- [ ] **Step 2: Confirm what already works.** `revoke_approval_rounds_for_run` is run-keyed and lives on the long-lived controller (`:4052-4145`) — verify a cross-turn child's approval cards still revoke correctly, with a test.
- [ ] **Step 3: Implement the decisions.** **Step 4: Gate + Commit.**

---

### Task 7: Docs, battery, live verification

- [ ] **Step 1: Docs** — `Docs/User_Guide/console/agent-runs-and-tools.md`: what it means for an agent to keep working after a reply finishes, how the user sees it, what bounds it, and what happens on app restart (`reconcile_orphaned_runs` sweeps `running` rows to `error` once per process, lazily on first bridge construction — so a child killed by a restart shows as errored, not silently gone).
- [ ] **Step 2: Battery** — `pytest Tests/Agents/ Tests/Chat/ Tests/DB/test_agent_runs_db.py -q` and `pytest --collect-only -q | tail -2`, READ every count. Compare failures against a pristine `origin/dev` checkout **at your branch's own base** before attributing any to this PR — comparing against current upstream has already produced one false regression this programme.
- [ ] **Step 3: Live verification** (`backlog/docs/lessons-live-verification.md` binds you) — tmux, scratch `TLDW_CONFIG_PATH`, never the live config; a working repo-root key (`openrouter-api-key.txt` 401s). Capture panes for: a child still running when the reply completes; that child reaching a terminal status afterwards with real content; sending a new user message while it runs and confirming it is NOT superseded; and an app restart leaving its row `error` rather than `running`.
- [ ] **Step 4: Backlog close-out + commit.**

---

## Deliberately NOT in this PR (→ PR 3a-2)

- **Auto-wake on completion + cross-conversation notification.** Spec §3
  invariant 5 was corrected on 2026-08-11: a finished child **wakes its
  supervisor** rather than queuing for the user's next message in that
  conversation, and the user-facing notification must be reachable from
  wherever they are. The premise behind the old "no auto-wake" ruling — that
  the user is watching one conversation — is false for this application,
  which is built around concurrent conversations, workspaces, watchlists and
  schedules and hopping between them.

  Two notes for that PR, from the survey: a transcript SYSTEM row **does not
  reach the model** (`console_chat_controller.py:10110-10113` drops everything
  but USER/ASSISTANT), so delivery must use the system-prompt fold, an
  evidence-style prefix, or the `turn_bundle_block` append at
  `console_agent_bridge.py:2205-2226`. And the injected notice must be
  unambiguously marked **not user input** — Claude Code's own notifications
  carry that warning explicitly, which is a hazard worth inheriting the fix
  for rather than rediscovering.

  This PR must not build anything that makes auto-wake harder: in particular,
  Task 6's audit should record what a completing cross-turn child can still
  reach, since that is what 3a-2 will wake the supervisor through.
- `send_to_agent` + mailboxes. Nothing today injects into a running loop: `messages` is a local of `run_agent_loop` (`agent_runtime.py:519`) and `LoopDeps` carries no message-mutation callable. The only external in-flight influence is cooperative cancellation.
- Finished-agent continuation (`resumed_from_run_id`).
- The notification chip (spec §7) — `ConsoleApprovalsChip` (`console_status_chips.py:53`) is the precedent.
