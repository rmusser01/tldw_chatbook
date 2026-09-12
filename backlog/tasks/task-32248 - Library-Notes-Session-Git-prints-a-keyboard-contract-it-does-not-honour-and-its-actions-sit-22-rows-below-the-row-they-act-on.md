---
id: TASK-32248
title: >-
  Library Notes Session Git prints a keyboard contract it does not honour and
  its actions sit 22 rows below the row they act on
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
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
- [ ] #1 Opening Session Git after 'Trust and check status' focuses the status list
- [ ] #2 Up/Down, Tab and Enter do what the panel's own line says they do, starting from the state the panel opens in
- [ ] #3 Stage and Commit are reachable by keyboard and render adjacent to the row they act on rather than at the floor of the pane
- [ ] #4 Covered by a test pinning the panel's entry focus after Trust, and a keyboard-only stage
<!-- AC:END -->
