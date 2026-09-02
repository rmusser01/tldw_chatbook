---
id: TASK-28004
title: >-
  Library media list - Escape from Reader focuses the filter input, swallowing
  Down/Enter
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
updated_date: '2026-09-02 04:53'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Originally filed as a cursor/marker desync (Enter opened a different row than the visibly marked one). Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). That desync is GONE - the return-receipt focus machinery keeps marker and selection in sync once the list has focus, and moving the selection auto-loads the item in the Reader. Residual defect: Escape from the Reader (footer "esc focus Items") lands keyboard focus on the "Filter media" INPUT above the rows, so the natural next keystrokes are swallowed - typed characters land in the filter, Down does nothing until Tab moves focus to the rows. Fix direction: land focus on the item ROWS (ideally the currently-loaded row) instead of the filter input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Escape from the Reader, Down/Up immediately move the list selection (no keystroke swallowed by the filter input)
- [ ] #2 Focus lands on the loaded item's row so the next Down selects (and auto-loads) the adjacent item
- [ ] #3 A regression test pins the Escape-then-Down path
<!-- AC:END -->
