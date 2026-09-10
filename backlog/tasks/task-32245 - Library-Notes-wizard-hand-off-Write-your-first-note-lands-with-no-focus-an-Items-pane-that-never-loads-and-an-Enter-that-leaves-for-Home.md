---
id: TASK-32245
title: >-
  Library Notes wizard hand-off: Write your first note lands with no focus, an
  Items pane that never loads, and an Enter that leaves for Home
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - wizard
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Regression, new since `c4a7b1911f`: the button was added by `0b1f67aef7` (task-32140, PR #2538). It is the right idea -- a wizard that hands a first-timer a destination instead of a dashboard -- executed wrong, and it fails on the first keystroke at the single highest-intent moment in onboarding.

Reproduced by the reconciling parent on a clean fresh profile: after the click the middle ~130 columns read "Loading local Library sources..." and still did at t = 113 s (`R/caps/52`, `53`), focus is parked on nothing, the footer reads `esc back to notes`, and the first Enter navigates to Home (`R/caps/54`). The same view reached via `Ctrl+N` parks focus on `[]Blank note`, reads `enter create note`, and does not stall (`R/caps/57`). Both assessors saw it independently.

Cause PROVEN in part. The focus step exists only on the rail-press route: `library_screen.py:20561-20570` (`if row_id == LIBRARY_ROW_CREATE_NOTE and self.is_mounted: call_after_refresh(self._focus_library_note_control, "#library-notes-create-blank")`), which `action_library_notes_new` reaches through `_select_library_rail_row`; the wizard route resolves only a target row id at `library_screen.py:10042-10044`. The stall is PROVEN at `library_browse_route_swap.py:177-183`: the list child stays the "Loading local Library sources..." Static while `screen._library_loaded` is False and no lookup error is set, and this route never flips it. The exact divergence point between the two routes is INFERRED.

Why the pinning test did not catch it: `test_wizard_exit_route_notes_navigates_to_library_new_note` asserts only the `NavigateToScreen` message against a **mocked receiver**, so neither focus nor the Items pane is exercised. A replacement must drive the real route through to the mounted canvas.

Fix: route the wizard exit through the same `_select_library_rail_row` entry the `Ctrl+N` binding uses.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Arriving at Notes from the wizard's 'Write your first note' parks focus on Blank note and shows `enter create note` in the footer, exactly as `Ctrl+N` does
- [ ] #2 The Items pane resolves on that route: no 'Loading local Library sources...' remains once the screen has settled
- [ ] #3 The first Enter on arrival creates a note and never navigates to Home
- [ ] #4 Covered by a test that drives the real wizard exit route through to the mounted Notes canvas -- not a mocked receiver -- and asserts both the focused control and the loaded list
<!-- AC:END -->
