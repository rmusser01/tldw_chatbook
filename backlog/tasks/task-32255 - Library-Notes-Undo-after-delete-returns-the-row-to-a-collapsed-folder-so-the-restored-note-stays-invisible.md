---
id: TASK-32255
title: >-
  Library Notes Undo after delete returns the row to a collapsed folder, so the
  restored note stays invisible
status: Done
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
- [x] #1 Undo expands the restored note's folder and reveals the row
- [x] #2 Selection lands on the restored row
- [x] #3 Covered by a test whose starting state has the target folder collapsed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce headlessly: rerun the 32124 undo test with the target folder COLLAPSED at start.
2. Trace why the locator's tree_expanded_ids update does not reach the render.
3. RED test: undo restores into a collapsed folder -> row visible in projection + selected.
4. Fix at the seam every restore routes through; verify live (delete -> Undo on a folder row).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What reproduced, and what did not.** The reported case — delete a note
from a folder, collapse the folder, press Undo — was driven live at 235x52
on a seeded profile at dev 4a14b3f36f and the row came back VISIBLE and
selected (`wave3-caps/list-tree/11-undo-collapsed-before.txt`, with the
selection colour confirmed in `11b-undo-collapsed-colour.txt`). The
locator already expanded the ancestors on that path, and the headless
32124 test passes with a collapsed start too. So the symptom the critique
saw is not reachable on the happy path here.

**The hole that is real.** The locator banked every expansion until BOTH
the folder path and the placement under it had answered, and returned
False from three bail-outs before applying any of it. One of those
bail-outs is reachable exactly when the critique saw the symptom: the
placements branch is marked stale by the restore's own reconcile and then
fails to re-page. The note is back, the count moves, and the folder stays
shut with nothing on screen saying why — the reported symptom, from a
cause the critique could not trace. Pinned by
`test_undo_opens_the_restored_folder_even_when_its_reload_fails` (RED:
"the restored note's folder stayed shut").

**Approach.** Each bail-out that runs while the navigation is still
current now opens the ancestors it did confirm. The first cut — expanding
inside the loop — broke four abandoned/superseded-locator pins, which
require a superseded locate to leave NO trace; the reveal is therefore
scoped to the two failure exits and the success exit, all of them past a
`current()` check.

**Coverage.** The 32124 undo test is now parametrized over a COLLAPSED
starting state (AC#3) and asserts the folder ends open (AC#1) and the
selection lands on the restored row (AC#2). Live after the fix:
`25-undo-collapsed-after.txt` (vault reopens, `Markdown showcase · now`
visible and selected).

**Files.** `tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_notes_wave_list.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
