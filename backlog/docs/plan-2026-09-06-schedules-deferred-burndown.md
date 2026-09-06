# Schedules deferred-tasks burndown (round 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Close the two tasks deferred from the 2026-09-05 close-out burndown (PR #2454): 31415 (post-transfer history-identity bridge) and 31419 (shell-chrome floor measurement).

**Spec:** the two backlog task files ARE the spec (`task-31415`, `task-31419`); ledger trail for 31415 in `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` §9/§12.

## Global Constraints
- NEVER `git stash`; baselines via `git show <rev>:<path>` or throwaway detached worktrees, removed after.
- FOREGROUND pytest only; tmp_path DBs; exact pasted tail lines from the final run.
- Diagnostics pin is a SCRIPT (`scripts/check_persistent_diagnostic_inventory.py --write` + commit JSON), not a pytest.
- Live app measurement uses a scratch profile (`TLDW_CONFIG_PATH`) on the real shell; never the real user profile.

### Task 1 (=31415): Bridge automation run/result history across a post-transfer identity change
One shared identity-resolution seam INSIDE `Scheduling/db/scheduled_tasks_db.py` (per AC#3): `list_automation_runs` (:2265) and `list_automation_results` (:2424) currently filter on a single `definition_id`; after `adopt_server_definition_identity` (:2913) a definition answers to two ids (local `id` + `server_id`), and `upsert_automation_results_from_server` (:2989) stores server ids verbatim. Resolve the alias set once (both directions: given either id, find the row and collect {id, server_id}) and filter `IN (aliases)` — the 6 workbench call sites then need no changes. Model: `index_definitions_by_id` (`unified_rows.py`) indexes under both id spaces. AC#4 is the regression gate: a never-transferred definition returns byte-identical rows and ordering. AC#5 (to-local direction): bridge if the data is local, else record why it is server-side work and name the gap in `Docs/User_Guide/schedules.md`. Commit `fix(scheduling): bridge run/result history across post-transfer identity change (31415)`.

### Task 2 (=31419): Reclaim narrow-terminal rows from the app shell — measure, classify, decide
A measurement exercise with a decide-and-close mandate. Re-measure the 13-row app-shell chrome figure IN THE REAL SHELL at 80x24 (tmux, scratch profile), record per element (AC#1); classify each element reducible / conditionally-reducible-at-floor / deliberate with reasons (AC#2). If a clean shell-level reduction exists, apply it at the shell so every destination benefits (AC#3) and prove Schedules' floor test passes unchanged (AC#4). If not, AC#5: record no-reclaim as the answer and close. Findings go in the task file; any measurement artifacts in the SDD workspace. Commit `docs(shell): narrow-terminal chrome measurement + ruling (31419)` (or `fix(shell): ...` if a reduction ships).

## After
Final review → PR `fix(scheduling): deferred burndown (31415, 31419)` → in-loop merge → mark both Done.
