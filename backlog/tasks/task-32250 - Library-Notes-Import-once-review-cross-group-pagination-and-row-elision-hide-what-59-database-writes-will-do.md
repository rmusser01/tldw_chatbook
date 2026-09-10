---
id: TASK-32250
title: >-
  Library Notes Import once review: cross-group pagination and row elision
  hide what 59 database writes will do
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found independently by both assessors, and it survives #2540's row-geometry fixes -- those did land and are pinned by name (`test_both_relationships_are_buttons_under_their_own_descriptions`, one-line rows with adjacent controls, >=15 rows at 235x52, `Skip all`/`Create all` per group). This is the residual, and it is the half that matters at the moment of approval.

Page 1 of 3 is 23 `vault/Archive/Archived note NNN.md` rows. Groups are split across page boundaries, so the counts change meaning per page (`New (23)`, `New (22)`, `New (13)`) and **the total never appears before you approve 59 database writes**. Every New row ends `. ke...`, hiding the keywords and link count the guide says the row exists to convey; the 120-character-filename row loses its entire outcome clause. Deciding what to skip requires holding page 1's 23 Archive rows in mind while reading page 3. Meanwhile the review -- the only thing being done -- gets about 120 of 235 columns while the Notes list it is not about keeps its share.

Partially fixed prior P2: task-32125 and task-32134 covered the chooser and the source-change half. Cause INFERRED from behaviour; not re-traced by the parent. Captures: C 30/33/34, D 25/26/27.

Fix direction: page within groups, collapse a long uniform run to one summary row with a disclosure, order Unsupported/Empty first, and give the review the full pane width while it owns the pane.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The total item count is visible before the approve action, not only per page
- [ ] #2 A group is never split across pages: the count shown on a page means the same thing as the count in its header
- [ ] #3 Every review row states the resulting title, its keywords and its link count without elision at 235x52
- [ ] #4 A long uniform run collapses to one summary row with a disclosure rather than 23 near-identical rows
- [ ] #5 'Next page' on the last page is visibly distinct from an active pager
- [ ] #6 The review owns the full pane width while it is the task in hand
- [ ] #7 Covered by a test for the un-elided row at 235x52 and a test for group-contained pagination
<!-- AC:END -->
