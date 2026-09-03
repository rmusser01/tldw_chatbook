---
id: TASK-28010
title: >-
  Library media Reader - content box capped at 18 rows while the pane has empty
  space
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
updated_date: '2026-09-02 04:53'
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
Originally filed against the old stacked-column viewer (analysis below an 18-line content porthole). Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). The stacking half is FIXED - the Reader now has Read/Analysis/Highlights/Info tabs. The height half remains: the content box is hard-capped at max-height: 18 (css/components/_agentic_terminal.tcss ~3440 block, mirrored in css/screen_agentic_library.tcss ~2350) while roughly 19 rows sit EMPTY between the tab row and the content box on a 52-row terminal (live capture: content occupied pane rows 32-49). A two-hour transcript is read through an 18-row porthole below a large blank region.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The analysis is readable without scrolling past the content block
- [ ] #2 Content height scales with terminal height instead of a fixed 18 rows
- [ ] #3 The full transcript remains reachable and scrollable
<!-- AC:END -->
