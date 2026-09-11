---
id: TASK-32248
title: >-
  Library Notes Session Git prints a keyboard contract it does not honour and
  its actions sit 22 rows below the row they act on
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 12:00'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
  - file-notes
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This is where the two assessors disagreed, and the re-test splits them.

D is right that the flow completes: the reconciling parent staged and committed end to end by mouse against real git -- `M  Daily/2026-09-07.md` after Stage, then `bd746be reconciler edit from Chatbook` with a clean working tree (`R/caps/34`, `37`).

C is right that the panel is unreachable as it advertises: from the state the panel opens in after "Trust and check status", `Down` -> `Tab` -> `Enter` left `git status` completely unchanged (`R/caps/31`-`33`) while the panel's own line reads `Up/Down select . Tab actions . Enter run . Esc back`. C concluded "no stage or commit action reachable" because the actions render at rows 44-47 of a 52-row pane, about 22 rows below the file row and below the fold of attention, and every key C tried did nothing.

Session Git is otherwise the best-designed flow on the screen -- trust modal naming the real risk, status list, pre-commit disclosure of parent sha, identity, hook policy, signing status and blast radius -- which is exactly why a keyboard contract it does not honour costs so much: it is what makes Folder files safe on a real repository.

Cause INFERRED: reproduced across four input attempts; the focus chain was not traced. `Tests/UI/test_library_file_notes_git.py` pins focus in several sub-states (`test_medium_commit_review_transition_reliably_focuses_edit`, `test_return_to_commit_list_keeps_result_until_called_then_focuses_row`) but nothing pins the panel's entry focus after Trust.

Fix: focus the status list when the panel opens -- it is the only actionable thing on it -- and dock Stage/Commit to the selected row rather than to the pane floor.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Session Git after 'Trust and check status' focuses the status list
- [x] #2 Up/Down, Tab and Enter do what the panel's own line says they do, starting from the state the panel opens in
- [x] #3 Stage and Commit are reachable by keyboard and render adjacent to the row they act on rather than at the floor of the pane
- [x] #4 Covered by a test pinning the panel's entry focus after Trust, and a keyboard-only stage
<!-- AC:END -->


## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live at 235x52 against a real git vault: Trust, then Down/Tab/Enter, and check `git status`.
2. Trace the focus chain: which control holds focus after Trust, and why.
3. RED tests: the first ready status focuses the row list; Tab from the list reaches Stage; Stage renders within a few rows of the row it acts on; a later refresh does not steal focus.
4. Fix: request the panel's existing settle-focus on the first ready status; bound `#file-notes-git-rows` so the actions stop rendering at the pane floor.
5. Live GREEN: keyboard-only stage and a real commit, verified with `git log`.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
The re-test split the assessors and the cause turned out to be both halves
of the same thing.

**Focus (AC#1/AC#2).** The trust button hides when trust lands, so
`_update_actions`'s `_repair_hidden_focus` rescued focus onto **Refresh** --
where Down does nothing, Tab reaches the list, and Enter runs nothing. That
is exactly what C measured. `render_status` now requests the panel's OWN
existing settle-focus (`_commit_list_focus_pending`, the machinery
`return_to_commit_list` already used) on the FIRST ready status, guarded by
`_focus_is_inside` so a status landing while the user is elsewhere never
pulls focus, and by `was_ready` so a Refresh or a post-stage re-render never
yanks it back.

**Distance (AC#3).** `#file-notes-git-rows` was `height: 1fr`, so the list
swallowed every spare row of the surface and the actions for the SELECTED
row rendered at the pane floor -- live capture: row 20 vs row 44 at 235x52.
Bounded to 12 cells (six two-cell rows) the list scrolls its own overflow
and Stage lands 3 rows under the row it acts on. Trade-off: a vault with
many session changes now scrolls inside the list instead of down the pane;
that is the price of keeping the actions adjacent, and the bulk actions
exist for the many-file case.

Live GREEN: Trust -> Down -> Tab -> Enter staged README.md (`git status`:
`M  README.md`), and the keyboard journey reached a real commit --
`git log`: `e69848b w3 round2 commit`, clean tree. (That is the SHIPPED
code's run; the first cut's `5da59ec` was produced by code changed twice
afterwards -- review F9.)

Files: `Widgets/Library/library_file_notes_git_panel.py`,
`Tests/UI/test_library_file_notes_git.py`,
`Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
