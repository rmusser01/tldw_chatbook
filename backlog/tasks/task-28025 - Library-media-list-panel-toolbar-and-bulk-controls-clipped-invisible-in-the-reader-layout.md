---
id: TASK-28025
title: >-
  Library media list panel - toolbar and bulk controls clipped invisible in the
  reader layout
status: Done
assignee: []
created_date: '2026-09-02 04:54'
updated_date: '2026-09-02 06:36'
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
- [x] #1 All list-panel toolbar controls are visible and clickable at the default 3-column layout widths
- [x] #2 Select-mode bulk controls are visible and clickable (wrap, stack, or overflow-menu strategy)
- [x] #3 A regression test pins toolbar control visibility at the reader-layout panel width
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the media list panel's ds-toolbars used default (content-sized) button widths, so in the 3-column reader layout (Items panel ~38 cols) the buttons after Export overflowed the panel border - Trash at x=77+16=93 vs panel-right=80, clipped off-screen and unclickable (headless region test reproduced this exactly). Fix mirrors the shipped conversations-canvas rule: #library-media-canvas > .ds-toolbar is width:100% and its .library-canvas-action / .library-toolbar-count children are 1fr min-width:0, so all controls share the panel width and stay on-screen (labels truncate rather than clip). Same rule fixes the select-mode bulk bar. Edited the component source (_agentic_terminal.tcss) and regenerated the bundle via build_css. Distinct from task-15140 (old full-width overflow below 110 cols) - this is the fixed-narrow reader panel at any width. Test: test_media_toolbar_actions_fit_the_items_panel_at_wide_width (region math, both toolbar and bulk bar). Files: css/components/_agentic_terminal.tcss, css/tldw_cli_modular.tcss (generated), Tests/UI/test_library_media_side_by_side.py.
<!-- SECTION:NOTES:END -->
