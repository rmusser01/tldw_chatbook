---
id: TASK-26035
title: Filesystem checkpoints with per-turn baseline and rollback
status: Done
assignee: []
created_date: '2026-08-31 15:47'
updated_date: '2026-09-02 06:20'
labels:
  - agents
  - workspaces
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no way to undo a turn's file changes as a unit. Verified on origin/dev: change review offers per-file revert and undo-all (UI/Screens/change_review_screen.py:1347-1348) over turn-scoped snapshots (Workspaces/change_turn_tracker.py:172,315,396 with rows in DB/AgentRuns_DB.py:321-348), but there is no time-indexed baseline a user can roll back to - and no way to ask what the tree looked like three turns ago. Hermes snapshots the working tree before file-mutating tools each turn into a shadow git store, with list, diff, session-diff and a safe restore plan. Chatbook already has the shadow-repo mechanism this would build on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A restorable checkpoint is captured per turn before file-mutating tool calls, reusing the existing shadow-repo snapshot mechanism
- [ ] #2 Checkpoints are listable with enough context to choose one (turn, time, what changed)
- [ ] #3 Restoring produces a preview of what will change before it is applied and requires confirmation
- [ ] #4 Restore refuses rather than proceeding when the working tree has uncommitted changes it would destroy, unless the user explicitly overrides
- [ ] #5 Checkpoints are pruned by a documented retention and size bound
- [ ] #6 Checkpoint storage stays out of the user's own git history - it never creates commits, branches or stashes in the project repo
- [ ] #7 Files outside the allowed roots are never captured or restored
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
SEAM ANALYSIS (2026-09-01, for a live session -- this restores USER FILES, so it must be built + verified with the app running, not headless):
Build on the existing shadow-repo (Workspaces/change_tracking.ShadowRepo): snapshot(message, force_paths)->sha, tip(), changed_files(base,end)->[ChangedFile], diff_text(base,end,path), file_bytes(commit,path), restore_paths(commit,paths). Per-turn baselines already captured by Workspaces/change_turn_tracker.ChangeTurnTracker (B/E snapshots; the B tip IS the pre-turn checkpoint) with rows in DB/AgentRuns_DB.py:321-348.
1. AC#1 capture: reuse the existing B snapshot per turn before file-mutating tools (already happens); expose those sha+turn+time as 'checkpoints'.
2. AC#2 list: enumerate checkpoints with turn/time/changed-file-count (changed_files against the prior tip).
3. AC#3 restore PREVIEW+confirm: a PURE restore-plan builder (given target sha + current tree via changed_files) -> the set of files that would change + a diff preview; require explicit confirmation before calling restore_paths. This is the testable core.
4. AC#4 refuse-on-dirty: before restore, detect uncommitted changes in the working tree that the restore would destroy; refuse unless an explicit override flag is passed.
5. AC#5 retention: prune checkpoints by a documented count/size bound (pure policy fn, testable).
6. AC#6 out-of-user-git: ShadowRepo already stores in a separate shadow store (never commits/branches/stashes in the project repo) -- verify + assert.
7. AC#7 roots-only: ShadowRepo already scopes to allowed roots -- verify + assert.
Testable now (pure): restore-plan builder, dirty-tree refusal decision, retention/prune policy. Impure/live: the actual snapshot/restore + the change-review UI surface. DEFERRED to a live session for safe verification of the restore path.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REJECTED by owner (2026-09-02): covered enough by existing surfaces — /rewind rolls the conversation back, and Change Review already offers per-file revert + undo-all over per-turn snapshots (change_turn_tracker B/E shadow-repo snapshots). The residual (one-shot tree restore to N turns ago + retention) is a nice-to-have with a destructive-restore risk profile that outweighs it. The shadow-repo seam analysis in this file stands if ever revisited.
<!-- SECTION:NOTES:END -->
