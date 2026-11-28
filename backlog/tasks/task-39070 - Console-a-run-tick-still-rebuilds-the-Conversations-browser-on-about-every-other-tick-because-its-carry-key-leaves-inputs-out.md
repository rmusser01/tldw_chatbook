---
id: TASK-39070
title: >-
  Console: a run tick still rebuilds the Conversations browser on about every
  other tick, because its carry key leaves inputs out
status: To Do
assignee: []
created_date: '2026-11-05 16:20'
updated_date: '2026-11-06 18:31'
labels:
  - console
  - performance
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36571.1 stopped the projection building the workspace context (it reads the label alone) and lets a run tick reuse its predecessor's build once while a carry key matches: the TASK-22201 fingerprint, with the run status read as the run markers and attention facts the build shows. The reuse is capped at once in a row, because the key leaves inputs out (relative ages and date buckets, stars, manual unread marks, activity receipts, appearance, folder availability and its refresh age, persisted-rows and appearance cache ages, archive states, sub-agent and progress counts, collapse preferences, rail height, session character and hand-off fields, the workspace conversation search), and TASK-36571.1's AC#3 allows those one poll interval behind the base build. So the Conversations tray push still rebuilds on about every other tick of a send window, and the build time per window fell by about a third to a half rather than the half TASK-36571.1's AC#2 asked for (see its notes for the paired figures).

Reusing an unchanged build for longer is exact only if every input the build reads is either in the carry key or changes only through a path that rebuilds and pushes outside the run tick. Today nothing enumerates those inputs, and a field-level census of consecutive builds in a send window showed several changing without a fingerprint change (row counts after a persisted-rows refresh, manual unread marks after their loader lands). The owner decided (2026-11-05) to merge TASK-36571.1 as a partial and keep its AC#2 target rather than restate it at the measured cut, so this task carries the rest of that target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every input the workspace-context build reads is either part of the carry key or changes only through a path that rebuilds and pushes outside a run tick, and a deterministic inventory test fails when the build starts reading an input that is neither
- [ ] #2 A run tick reuses an unchanged build for as long as its carry key is unchanged, and the run tick's workspace-context build time per send window at least halves against dev, live, real provider, paired, at a stated chat length
- [ ] #3 What the rails show is never staler than one poll interval behind dev for every inventoried input, with a deterministic test per input class
- [ ] #4 A send reaches the provider no later than on dev (Enter to the provider request, live, paired and interleaved, at least 30 sends per arm including first sends)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-06, from TASK-36571.1's review (for the inventory, AC#1 and AC#3):
- The attention rows (`_conversation_attention_rows`) also read `activity_receipts.unseen_snapshot()` and the manual-unread marks, which the carry key leaves out. Today a receipt is published in the same step as its run's marker change, and a manual-unread landing pushes on its own. Adding the receipts' `projection_generation` (a cheap int) to the key would make that input exact.
- 'At most one tick later' assumes a later tick. The poll's final paint can itself be a reuse (key unchanged since an end-of-run sync), and then nothing rebuilds. No live case was found: every uncovered input traced (unread acknowledgement, sub-agent counts, folder availability on change, persisted-rows landing) pushes on its own. The inventory test should cover this case.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Filed as TASK-38950 on this task's branch (perf/task-36571.1-browser-state-rebuild, created 2026-11-05 16:20 UTC). While that branch was unmerged, the round-8 follow-up "Every return to the Console restyles the whole screen, and a vertical breakpoint flip restyles it again" landed on dev first as TASK-38950 (8f32a29bab, 2026-11-05 18:06 -0800), and the rebase onto dev e8eeae7db9 failed preflight's duplicate-id check. Per the TASK-19601 owner rule (the older arrival keeps the id), the task already on dev keeps TASK-38950 and this one, not yet on dev, renumbers. TASK-39070 was unused on every ref, in every local worktree and on origin/dev (the highest id seen was 39040). The owner's 2026-11-05 decision on TASK-36571.1 ("TASK-38950 (the carry-key cache) does the rest") refers to this task.
