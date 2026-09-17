---
id: TASK-32656
title: >-
  Library Notes: the two reviews share a collapse threshold but not a run order,
  so interleaved folders collapse on one screen and not the other
status: To Do
assignee: []
created_date: '2026-09-15 20:15'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from wave-5 group 6's fix round 2 review (task-32625). PRE-EXISTING on dev -- not introduced by that task, and out of its scope.

Import once and lasting sync now agree on WHEN a run of skipped files collapses: both use UNIFORM_RUN_MIN (8) and both key a run on the folder. What they do not share is the ORDER the runs are computed over. uniform_runs is a groupby, so it only ever joins CONSECUTIVE members -- which makes the sort that precedes it part of the contract.

LibraryNotesAddFromFilesCanvas._compose_review_rows sorts each group's members by relative_path before calling uniform_runs. LibraryNoteImportCanvas._compose_review sorts only by classification (the group order the pager plans against, task-32250) and never by path within a group.

So with two folders' skips interleaved at eight files each, the sync review collapses both into folder summaries and Import once collapses neither -- the same content, the same threshold, two different screens again. The plan's own ordering is what decides it, so a vault whose planner emits shuffled paths (the sync side's comment at _compose_review_rows says a plan orders its actions by binding id, a digest) hits this rather than it being a corner case.

Not fixed in wave 5 group 6: that task's AC was the threshold and the unit, both of which now agree, and Import once's ordering is load-bearing for its pager.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both reviews compute their runs over the same ordering, or the difference is recorded with the reason Import once's pager needs its own
- [ ] #2 A pin composes both canvases from one fixture with two folders' skips INTERLEAVED at the threshold and asserts they collapse the same way
- [ ] #3 The pager's page-fill assumption still holds under whatever ordering is chosen (task-32250)
<!-- AC:END -->
