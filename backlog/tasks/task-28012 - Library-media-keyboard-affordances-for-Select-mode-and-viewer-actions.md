---
id: TASK-28012
title: Library media - keyboard affordances for Select mode and viewer actions
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Originally covered both the old viewer's five-button action row (now obsolete - the Reader has a More overflow menu) and the list's Select mode. Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). The Select-mode half stands: Space on a focused row toggles nothing in either normal or select mode, no key is advertised for entering Select mode or toggling rows, and the counter stays "0 selected". Related but distinct rendering defect (clipped-invisible Select/Trash/bulk buttons) is tracked separately - see the toolbar-clipping task filed from the same run. Scope here: a keyboard path to enter Select mode, toggle rows (Space), and reach the bulk actions, advertised in the footer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Row selection can be entered and toggled from the keyboard, advertised in the footer
- [ ] #2 Viewer actions have bound keys shown in the footer or help panel
- [ ] #3 Existing mouse paths are unchanged
<!-- AC:END -->
