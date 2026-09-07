---
id: TASK-1971
title: 'Change review: B/E turn snapshots around agent runs + change_snapshots schema'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
  - agents
  - db
dependencies:
  - TASK-1970
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the turn protocol: baseline snapshot B at run start (skipped when clean — B = previous tip), end snapshot E at run end; B == E means no changes. Records go in a new AgentRunsDB `change_snapshots` table (run_id, root, baseline_sha, end_sha, files_changed, adds, dels, reverted, tracking_error; schema bump per repo discipline). One row per (run, root) for multi-root workspaces. The FIRST snapshot of a root happens at root-registration time on a background worker — never as first-send latency. Failure posture: tracking never blocks the agent; a failed snapshot logs, stores tracking_error, and the run proceeds.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run that writes a file yields a change_snapshots row whose diff(B,E) matches disk truth exactly (property test against real git)
- [x] #2 A run that touches nothing yields NO row and no card
- [x] #3 A change made by a SCRIPT the agent ran (not write_file) appears in diff(B,E)
- [x] #4 Snapshot failure (e.g. git binary removed mid-session) stores tracking_error and the agent reply still completes
- [x] #5 Registering a root triggers its initial snapshot in the background; the first send performs no full-tree add
- [x] #6 Schema migration applies cleanly to an existing AgentRunsDB
- [x] #7 B runs in parallel with the model request and completes before the FIRST tool executes (asserted by ordering probe), so a send adds no user-visible snapshot latency
- [x] #8 A FAILED or cancelled run still records E and its row -- the half-finished edit set is reviewable
- [x] #9 An agent file-tool write to a .gitignore'd path (e.g. .env) appears in the turn's diff (force-add carve-out); a SCRIPT write to an ignored dir does not, and the limit is documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD in `Tests/Chat/test_change_turn_tracking.py`: tracker-level against real git (turn round-trip = disk truth, clean turn = no records, begin non-blocking + await gates via a latency-shimmed service, force-add carve-out incl. the script-to-ignored-dir negative, failure -> tracking_error records, never raises); DB-level (v2 file gains the table on open, record/read round-trip); bridge-level (gateway side-effect write mid-stream -> row; failed run still records E; ordering probe: baseline completes before the first real tool executes via a registry-injected tool).
2. `Workspaces/change_turn_tracker.py`: `ChangeTurnTracker.begin_turn(roots)` (one background thread snapshotting each root; per-root errors recorded, never raises), `TurnHandle.await_baseline()`, `end_turn(handle, touched_paths)` (force-add tool-touched within each root, E snapshot, B==E skip, per-root `TurnChangeRecord` incl. `tracking_error`), `tool_touched_paths(steps)` restricted to WRITE tools (read touches would force-add pre-existing ignored files and lie an "A" row).
3. `ShadowRepo.force_add(paths)` primitive in change_tracking.py (add -f under the lock).
4. `AgentRuns_DB`: `change_snapshots` table + index via the existing CREATE IF NOT EXISTS discipline (version row 3), `record_change_snapshot` / `change_snapshots_for_run` / `change_snapshots_for_conversation`.
5. Bridge: `change_tracker` ctor kwarg + `change_roots` run_reply kwarg; begin at entry (guarded), review hook wrapped so its first invocation awaits B, end_turn + row writes in the run_turn `finally` (E on every terminal path; a crash before run_id logs records instead of writing rows).
6. Controller: lazy `ChangeTurnTracker`, roots from the workspace registry's folder bindings for THIS run's workspace, passed into run_reply; registration hook = best-effort background initial snapshot after `add_folder_binding`.
7. Sabotage anything that passes first try.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`Workspaces/change_turn_tracker.py` (begin/await/end + `tool_touched_paths` + `initial_snapshot_in_background`), `ShadowRepo.force_add`, AgentRuns_DB v3 (`change_snapshots` via the CREATE-IF-NOT-EXISTS discipline + record/read methods), bridge wiring (ctor `change_tracker`, run_reply `change_roots`, review-hook wrap as the await-B gate, end_turn + row writes in the run_turn finally), controller roots via new `workspace_file_roots.folder_binding_roots` (ro INCLUDED -- scripts write where tools cannot; sandbox EXCLUDED -- app scratch is review noise), screen hands the bridge a tracker only when git is available, `add_folder_binding` fires the background initial snapshot.

**The tests taught the design something.** My first bridge write-tests fired their side effect during the FIRST provider stream -- racing the baseline thread. Warm process: the gateway wins, the write lands INSIDE B, and B==E hides it; cold process: baseline wins and the test passes. One test failed deterministically in-file and passed alone (bisected pairwise); the other passed only by cold-start luck. The window is real but production-empty: every writer, scripts included, is a tool behind the await-B gate -- only a non-tool writer racing the first token can hit it, which is spec §5's documented attribution limit. Characterized in the module docstring; tests restructured to write post-gate.

Two verification traps recorded: the registration hook's default-constructed service would have written shadow repos into the REAL app data dir from every registry test -- verified `get_user_data_dir()` resolves inside the per-test isolated XDG tree under pytest before trusting it. And my gateway fake was a sync generator against an async-generator contract, so all four bridge tests failed with empty runs until it matched the real `_ChunkGateway` shape.

Three sabotages, each failing exactly its test: await-B gate removed (ordering), force_add removed (carve-out), finally end_turn removed (records + failed-run).
<!-- SECTION:NOTES:END -->
