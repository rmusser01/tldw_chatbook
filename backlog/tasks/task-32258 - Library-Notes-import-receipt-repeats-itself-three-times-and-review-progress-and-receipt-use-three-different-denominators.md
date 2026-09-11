---
id: TASK-32258
title: >-
  Library Notes import receipt repeats itself three times and review progress
  and receipt use three different denominators
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:24'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On one 71-file vault import, within a single journey: the review counted **66 items**, progress reported **67 complete**, and the receipt reported **59 created + 8 skipped**. Three numbers for one import, none of them wrong on its own terms, all of them visible to the same user in the same minute. The receipt itself then states its outcome three times in three formats.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One denominator is used across review, progress and receipt, or each surface states what it is counting
- [ ] #2 The receipt states its outcome once
- [ ] #3 Covered by a test asserting the three surfaces agree for a single import
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run one real import end to end and compare review/progress/receipt numbers.\n2. Name the denominator on every surface and state the receipt outcome once.\n3. RED/GREEN test asserting the three surfaces agree for one import.
<!-- SECTION:PLAN:END -->
