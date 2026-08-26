# Supervisor Fleet PR 2b — Console Fleet Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the running fleet visible and inspectable in the Console rail — a summary line, per-agent rows with real live status, and drill-in — built as the first section of a reusable Inspector-style component.

**Architecture:** Close the reachability gap first (the UI cannot see the live `FleetCoordinator` today), then key live state per `run_id` instead of per conversation, then replace the single newline-joined Static with per-row widgets following `ConsoleRunInspector`'s structural-key + in-place-update pattern, fed through a coalescer.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest + pytest-asyncio (headless `run_test` harness).

**Spec:** `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md` §7 (the three-state Agents section, the Inspector direction, and the scope discipline: this PR ships the section component + Agents section ONLY — Changes/Sources/Workspace sections stay filed).

## Global Constraints

- Branch off **merged dev** (PR 2a must land first) into `.worktrees/fleet-pr2b`. Never git outside the worktree; never `git stash`; push after every task.
- pytest is the ONLY python entry point. A bare `python -c` importing `tldw_chatbook.config` triggers the app's config rewrite and has touched the user's LIVE config. Never read/write `~/.config/tldw_cli` or `~/.local/share/tldw_cli`.
- **Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss`** — it is generated. Edit `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate via `tldw_chatbook/css/build_css.py` (manifest `CSS_MODULES`). `check_bundle_sync.py` guards this; a hand-edit once silently shipped (TASK-395).
- Every new CSS class token must have a rule or an explicit `KNOWN_UNSTYLED` entry — `Tests/UI/test_css_class_coverage_contract.py` enforces it.
- **Rendered-geometry assertions, not DOM presence.** Unbounded-width Statics render invisible to headless queries while "present". Reuse `Tests/UI/test_console_parallel_runs.py`'s `_assert_widget_and_ancestors_displayed` (:825) and `_assert_painted_at_own_region` (:992) — the latter exists because `Widget.region` is UNCLIPPED, so only a compositor hit-test distinguishes "below the fold" from "visible".
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

### Verified seam map (survey against PR 2a's branch) — cite these, do not re-derive

| Seam | Location |
|---|---|
| Rail text builder (3-tuple) | `UI/Console_Modules/agent.py:354` `_console_agent_section_lines()` |
| **Sub-agent rows = ONE joined string** | `agent.py:464-466`, hard `[:60]` slice |
| Glyph map | `agent.py:457-463` |
| Fleet summary line | `agent.py:469-486`; counts from `console_chat_controller.py:1730` `fleet_summary_counts()` |
| The single Static | `left_rail.py:508-525` (`#console-agent-section-subagents`) |
| DOM apply + equality guard | `chat_screen.py:3960-4008`, guard at `:3976`, 0.2s tick at `:3958` |
| Click → drill-in | `chat_screen.py:18843-18846` → `agent.py:740-771` |
| `SubAgentSummary` (2 fields) | `console_agent_bridge.py:730-742` |
| **Live rows never get a real status** | `console_agent_bridge.py:2047-2049` — appended only on primary `STEP_SPAWN`, `status` left at its `"running"` default |
| Live snapshot writes | `console_agent_bridge.py:2020`, `:2095-2100`, `:2345-2350` |
| Historical (real statuses) | `console_agent_bridge.py:2854-2887` |
| Coordinator API | `Agents/fleet_coordinator.py` — `snapshot():216`, `drain_events():238`, `live_count():229` |
| **`drain_events()` has NO production consumer** | grep: definition + tests only |
| **Reachability gap** | coordinator lives on `AgentService._fleet` (`agent_service.py:508`); the service is a **local** in `run_reply` (`console_agent_bridge.py:2168-2181`), never stored on the bridge |
| `on_step` run_id accepted and DISCARDED | `console_agent_bridge.py:2026-2032` — its comment already names this PR's job |
| CSS source / bundle / guard | `components/_agentic_terminal.tcss` (agent rules `:3066-3084`) / `tldw_cli_modular.tcss` (generated) / `check_bundle_sync.py` |
| Row-widget precedents | `console_run_inspector.py:152-179` (structural key + in-place), `home_rail.py:88-128` (targeted patch), `console_workspace_context.py:537-567` (documented always-recompose deviation) |
| Coalescer precedent | `chat_screen.py:17866-17885` (scheduled flag + `call_after_refresh`) |
| Test harness | `Tests/UI/test_console_agent_controller.py:104-154`; size `(180, 48)` |

---

### Task 1: Expose the live fleet to the UI

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (store the per-run service/coordinator; add an accessor)
- Test: `Tests/Chat/test_console_agent_bridge.py` (append)

**Interfaces:**
- Produces: `ConsoleAgentBridge.fleet_snapshot(conversation_id) -> list[FleetHandle]` returning `[]` when no fleet is live. Keep the coordinator itself private — the UI must not mutate it.

**Why:** `AgentService` is a local inside `run_reply`, so nothing outside that call can see `_fleet`. Without this the panel can only render DB rows, which is exactly the staleness PR 2a's own live path already suffers.

- [ ] **Step 1: Write the failing test**

Model it on the bridge tests' existing construction. Assert: with a run in flight that reserved two handles, `fleet_snapshot(conv)` returns both with their real statuses; after the run completes it returns `[]` (or the terminal handles — pick one and pin it explicitly in the test name); for an unknown conversation it returns `[]` without raising.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_agent_bridge.py -v -k fleet_snapshot`
Expected: FAIL (no such method).

- [ ] **Step 3: Implement**

Store the live coordinator per conversation on the bridge for the duration of the run (set where the service is built at `console_agent_bridge.py:2168-2181`, cleared in the same `finally` that already tears the run down — find it, do not add a second teardown path). `fleet_snapshot` returns `coordinator.snapshot()` (already copies) or `[]`.
**Thread-safety:** the coordinator is written on the run's worker thread and read from the UI thread. `snapshot()` is lock-guarded, but the *dict holding it* is not — guard the store with a `threading.Lock`, or use a single attribute write (assignment is atomic under the GIL) and say which you chose and why.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_agent_bridge.py -q` → all pass, READ the count.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat: expose the live fleet snapshot to the Console UI" && git push
```

---

### Task 2: Key live state per run_id (real per-child status)

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`SubAgentSummary`, the `on_step` closure, the live snapshot writes)
- Test: `Tests/Chat/test_console_agent_bridge.py`

**Interfaces:**
- Produces: `SubAgentSummary` gains `run_id: str = ""`, `handle_id: str = ""`, and a real `status` that reaches terminal. The bridge's `on_step` stops discarding `run_id`.

**Why:** today a live row is appended once on `STEP_SPAWN` and its status is the `"running"` default **forever** — the panel would show every child as running until the historical path re-derives it after the turn. That is the single biggest correctness gap in the current surface.

- [ ] **Step 1: Write the failing tests**

Assert: a child that finishes `done` has a live `SubAgentSummary` with `status == "done"` **before** the turn ends; two concurrent children get distinct `run_id`s and their statuses do not cross; a child that errors shows `error`, not `running`.

- [ ] **Step 2: Verify failure**, then

- [ ] **Step 3: Implement**

Use Task 1's coordinator as the status source (it is the authority for live children; the DB is authority after the fact — spec §3 invariant 3). Rebuild the `subagents` tuple from `fleet_snapshot` on each snapshot publish rather than appending once. Keep the DB-derived historical path unchanged as the post-turn source.

- [ ] **Step 4: Gate** — `pytest Tests/Chat/ -q`, READ the count. **Also run `Tests/UI/test_console_agent_rail.py` and `Tests/UI/test_console_agent_controller.py`** — they assert on the current joined-string shape and may legitimately need updating; if so, preserve each assertion's meaning and say what moved.

- [ ] **Step 5: Commit**

---

### Task 3: The Inspector section component

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_inspector_section.py`
- Create: `Tests/UI/test_console_inspector_section.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` + regenerate the bundle

**Interfaces:**
- Produces: a reusable section widget — header (title + optional chevron + optional right-aligned summary), a body of rows, an optional "View all" tail. Rows are supplied by the caller as a list of value objects; the component owns layout and the structural-key/in-place-update discipline, not the data.

**Why a component:** spec §7 — Changes / Sources / Workspace sections follow. Ship the grammar once. **Scope discipline: this PR builds the component and the Agents section only.**

- [ ] **Step 1: Write the failing tests** — rendered-geometry, not DOM presence: header and each row have `region.width > 0` and `region.height > 0`; the summary is right-aligned within the header's region; collapsing hides the body but keeps the header painted; a row past the fold is caught by `_assert_painted_at_own_region`.

- [ ] **Step 2: Verify failure**, then

- [ ] **Step 3: Implement**

Follow `ConsoleRunInspector` (`console_run_inspector.py:152-179`): compute a structural key; when it is unchanged, patch rows in place and return; otherwise `refresh(recompose=True)`. Expose a `recompose_count` test seam exactly as that class does (`:150`).
**Read `console_workspace_context.py:537-567` before choosing** — that widget deliberately reverted an equality guard because skipping recompose broke click targeting on its rows. If your rows are clickable, either recompose or prove targeting survives the in-place path with a test that clicks a patched row.
Add CSS to the source module; regenerate; ensure `check_bundle_sync.py` passes and every new class has a rule (or a justified `KNOWN_UNSTYLED` entry).

- [ ] **Step 4: Gate** — the new file plus `Tests/UI/test_css_class_coverage_contract.py` and `Tests/UI/test_console_agent_tool_row_css.py`.

- [ ] **Step 5: Commit**

---

### Task 4: Agents section — three states

**Files:**
- Modify: `UI/Console_Modules/agent.py`, `UI/Console_Modules/left_rail.py`, `UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_agent_rail.py`, `Tests/UI/test_console_fleet_panel.py` (create)

**The three states (spec §7):**
1. **Summary line** (collapsed, the common case): glyph cluster + `N working` with `M done` right-aligned — the grammar from the owner's screenshots.
2. **Expanded rows**: one per child, **two lines** (line 1: glyph + agent name + elapsed; line 2: dimmed last-step summary, truncated). The rail clips single dense lines at default width (task-226). Scrollable/virtualized past a screenful; "View all" opens full run history.
3. **Drill-in**: the existing per-child transcript path, now reached by clicking a specific row rather than cycling.

- [ ] **Step 1: Write the failing tests** — one per state, rendered-geometry asserted; plus: clicking row *k* drills into child *k* (not the cycle behavior), and the summary line's counts match the coordinator snapshot.

- [ ] **Step 2: Verify failure**, then

- [ ] **Step 3: Implement**

Replace the single `#console-agent-section-subagents` Static (`left_rail.py:508-525`) with the Task 3 component. Retire the `[:60]` slice in favor of the component's own width handling. **Preserve the equality-guard contract** at `chat_screen.py:3975-3977` — `test_agent_section_sync_skips_repainting_an_unchanged_payload` pins it and must keep passing.
Replace the cycling drill-in (`agent.py:740-771`) with per-row click routing; follow `console_status_chips.py:53-72` (a row posts its own `Message`) rather than id-string matching in `chat_screen`.

- [ ] **Step 4: Gate** — `pytest Tests/UI/test_console_agent_rail.py Tests/UI/test_console_agent_controller.py Tests/UI/test_console_parallel_runs.py Tests/UI/test_console_fleet_panel.py -q`, READ counts.

- [ ] **Step 5: Commit**

---

### Task 5: Coalescing, cost rollup, and cancel

**Files:**
- Modify: `UI/Screens/chat_screen.py` (coalescer), `UI/Console_Modules/agent.py` (cost, cancel)
- Test: `Tests/UI/test_console_fleet_panel.py`

- [ ] **Step 1: Failing tests** — a burst of N fleet events produces ONE panel sync (assert via a `recompose_count`/sync-counter seam, the way `chat_screen.py:17866-17885`'s precedent is testable); per-child token spend appears in the expanded rows and the aggregate reaches the existing cost ticker; a Cancel action on a row cooperatively cancels that child and (PR 2a's guarantee) revokes its pending approval cards.

- [ ] **Step 2: Verify failure**, then

- [ ] **Step 3: Implement** — coalescer modeled on `_request_console_control_bar_sync` (scheduled flag + `call_after_refresh`), NOT a timer. Cancel routes through the coordinator's existing cancel path; do not add a second cancellation mechanism.

- [ ] **Step 4: Gate + Step 5: Commit**

---

### Task 6: Docs, battery, live verification

- [ ] **Step 1: Docs** — update `Docs/User_Guide/console/agent-runs-and-tools.md` (the panel, its three states, per-row cancel) and stamp with the sha of the final content commit.

- [ ] **Step 2: Targeted battery** (READ every count)

```bash
pytest Tests/UI/test_console_agent_rail.py Tests/UI/test_console_agent_controller.py \
  Tests/UI/test_console_fleet_panel.py Tests/UI/test_console_inspector_section.py \
  Tests/UI/test_console_parallel_runs.py Tests/UI/test_css_class_coverage_contract.py \
  Tests/Chat/ Tests/Agents/ -q
pytest --collect-only -q | tail -2
```
**Known pre-existing dev-baseline red (NOT yours):** `Tests/Chat/test_tool_output_disclosure.py` — 3 failures, verified identical on a pristine `origin/dev` checkout. Confirm the set is unchanged; a fourth is yours.

- [ ] **Step 3: Live verification** (per `backlog/docs/lessons-live-verification.md`)

tmux only; scratch `TLDW_CONFIG_PATH` (never the live config); a working repo-root key (`openrouter-api-key.txt` 401s on chat completions). Capture panes for: two children visible as separate rows with distinct live statuses; a row reaching `done` **during** the turn (this is Task 2's whole point); clicking a row drills into that child; cancel from a row leaves that child `cancelled` and its sibling unaffected; the summary line's counts match. Report honestly if a check cannot be completed.

- [ ] **Step 4: Backlog close-out + commit**

---

## Self-review notes (already applied)

- Spec §7 coverage: summary/rows/drill-in (T4), coalescer + cost + cancel (T5), section component for the later Inspector sections (T3). Changes/Sources/Workspace sections deliberately NOT in scope.
- The two hard prerequisites are T1 (the UI genuinely cannot reach the live coordinator today) and T2 (live rows never reach a terminal status), which is why they precede any widget work.
- `drain_events()` gets its first consumer here — if T5's coalescer uses it, note that the coordinator is currently rebuilt per turn, so unbounded event growth only becomes real in PR 3a.
