---
id: TASK-32255
title: >-
  Library Notes Undo after delete returns the row to a collapsed folder, so the
  restored note stays invisible
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:10'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual of task-32124. D: the rail count moves 61 -> 60 -> 61 and the `vault` folder row returns, but **collapsed**, so the restored row itself is never visible and the criterion "selection moves to the restored row" could not be confirmed (D cap 40). C saw the expanded-branch case work correctly (C cap 25).

So the fix landed for an expanded branch and not for a collapsed one -- the case where the user most needs visible proof that Undo did something, since the receipt is the only safety net the screen has.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Undo expands the restored note's folder and reveals the row
- [ ] #2 Selection lands on the restored row
- [ ] #3 Covered by a test whose starting state has the target folder collapsed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce headlessly: rerun the 32124 undo test with the target folder COLLAPSED at start.
2. Trace why the locator's tree_expanded_ids update does not reach the render.
3. RED test: undo restores into a collapsed folder -> row visible in projection + selected.
4. Fix at the seam every restore routes through; verify live (delete -> Undo on a folder row).
<!-- SECTION:PLAN:END -->
