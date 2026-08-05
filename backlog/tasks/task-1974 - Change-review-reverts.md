---
id: TASK-1974
title: 'Change review: per-file revert and Undo-all with user-edit guards'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - console
  - change-review
  - safety
dependencies:
  - TASK-1970
  - TASK-1971
  - TASK-1973
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore-to-B actions from the Review screen: per-file (`u`) and whole-turn Undo-all (`U`), both confirmed. Modified/deleted restore via checkout from B; CREATED files un-create via explicit guarded delete (`checkout B -- path` errors on a path absent from B); renames restore old and remove new. Guard: before any revert, each target's disk state is compared to E — files that differ (user or later turn edited them) are listed BY NAME in the confirm before anything is overwritten. Every revert takes a fresh snapshot and updates the row's `reverted` field. Runs under the per-root lock; per-file outcomes reported individually — partial failure is never silent.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Round-trip tests: create/modify/delete/rename each reverted back to exact B state on disk
- [x] #2 Un-create removes the file without touching neighbors; revert of a B-absent path never calls checkout
- [x] #3 Editing a file after E then reverting lists that file by name in the confirm dialog (sabotage: removing the guard fails the test)
- [x] #4 Undo-all on a turn whose file was ALSO changed by a later turn warns before clobbering
- [x] #5 After revert, a fresh snapshot exists and `reverted` reflects what happened
- [x] #6 A revert failure on one file reports that file and completes the rest
- [x] #7 Revert refuses with 'finish or stop the run first' while ANY run is active on the root (sabotage: removing the guard fails the test)
- [x] #8 Undo-all on a multi-root turn restores every root's files
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD the ENGINE first (`Workspaces/change_revert.py`, real git): `preflight` (per-path disk-vs-E comparison naming user-edited files; active-run refusal via an injected probe) and `revert_paths` (A -> guarded delete only when absent from B; M/D -> restore from B; R -> restore old + guarded-delete new; per-path outcomes, one failure never silences the rest; fresh "revert" snapshot; db `reverted` update).
2. `AgentRunsDB.update_change_snapshot_reverted(row_id, value)`.
3. Screen wiring: `u`/`U` bindings -> preflight -> `ChangeRevertConfirmModal` (lists edited-since files BY NAME, warns when a later turn touched the file) -> engine -> reload the turn; refusal copy when a run is active.
4. Sabotage: guard removed (edited-since), un-create calling checkout, refusal removed, partial-failure silenced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`Workspaces/change_revert.py` (engine: `preflight_revert`, `revert_paths`, `RevertRefusedError`, guarded un-create), `AgentRunsDB.update_change_snapshot_reverted` (appending merge — a second partial revert must not erase the first's record), provider seams (`preflight_revert`/`revert`/`run_active`), screen `u`/`U` actions + `ChangeRevertConfirmModal` (names edited-since files), per-path outcome reporting (failures named, never rolled up), post-revert reload from disk truth, and the production `run_active` probe reading the controller's live run state.

**Git is far more forceful than assumed — established empirically while building the failure-isolation test.** `checkout <sha> -- path` force-clobbers a non-empty DIRECTORY squatting at a file path (occupant deleted, rc 0) and tunnels through a FILE squatting at a directory path. Two "obvious" failure injections turned out to be successes; only POSIX permissions (read-only parent) actually make a restore fail as non-root. The isolation test now injects that, and the first-failure-aborts sabotage — which SURVIVED the original weak test via the no-exception branch — now fails it.

Also caught mid-wiring: `CONSOLE_ACTIVE_RUN_STATUSES` is chat_screen's own module constant, not console_chat_models' — a nested import from the wrong module would have crashed the opener at first use.

Four engine sabotages (refusal removed, naive-checkout un-create, guard blinded, first-failure-aborts) each fail exactly their test; the confirm/warning/refusal flows are pinned at the mounted screen. 40 passed across the engine, screen, and turn-tracking suites.
<!-- SECTION:NOTES:END -->
