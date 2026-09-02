---
id: TASK-28025
title: >-
  Library media list panel - toolbar and bulk controls clipped invisible in the
  reader layout
status: To Do
assignee: []
created_date: '2026-09-02 04:54'
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
Live 2026-09-02 on dev tip (media-ux-fixes worktree @ b7e89b6de), 235-col terminal: the Media list panel (~38 cols in the 3-column reader layout) clips its header toolbar after 'type: All types   Export...' - the Trash and Select buttons compose (Widgets/Library/library_media_canvas.py ds-toolbar Horizontal, ~376-421) but render past the panel border: invisible and unclickable, reachable only by Tab-walking focus onto invisible buttons and pressing Enter. In Select mode the bulk bar clips after 'Select all 3 shown' (Clear / Export selected / Delete selected all invisible; only a stray focus edge shows at the border). Distinct from task-15140 (old full-width toolbar overflow below ~110 cols): this happens at ANY terminal width because the panel column is fixed-narrow. Bulk actions and Trash are effectively mouse-unreachable for everyone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All list-panel toolbar controls are visible and clickable at the default 3-column layout widths
- [ ] #2 Select-mode bulk controls are visible and clickable (wrap, stack, or overflow-menu strategy)
- [ ] #3 A regression test pins toolbar control visibility at the reader-layout panel width
<!-- AC:END -->
