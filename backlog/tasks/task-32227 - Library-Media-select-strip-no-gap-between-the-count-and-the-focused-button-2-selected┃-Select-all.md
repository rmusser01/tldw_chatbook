---
id: TASK-32227
title: >-
  Library Media select strip: no gap between the count and the focused button
  ('2 selected┃ Select all')
status: Done
assignee: []
created_date: '2026-09-10 14:56'
updated_date: '2026-09-10 17:33'
labels:
  - library
  - media
  - layout
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The count and the adjacent focused button's heavy border touch. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 26.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One cell of padding between the count and the first action at every width
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing geometry test: one cell between #library-media-selected-count and #library-media-select-all
2. margin-right 1 on the shared .library-toolbar-count rule
3. Rebuild the CSS bundle; re-run the toolbar/multiselect pins
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One cell of right margin on the shared .library-toolbar-count rule in css/components/_agentic_terminal.tcss (regenerated screen_agentic_library.tcss committed alongside). Fixed on the class rather than inline on the Static, honouring the decision recorded above that widget in library_media_canvas.py: every canvas's counter is covered by one declaration, not a per-widget Python one-off -- so the Conversations counter and the bulk-delete receipt copy get the same cell. Pinned by a geometry test parametrised over 235x52 and 100x30 (first.region.x - count.region.right >= 1), which failed at BOTH sizes before the rule. Live: '2 selected ┃ Select all 2 shown ┃' at both widths, previously '2 selected┃ Select all'. Files: _agentic_terminal.tcss, screen_agentic_library.tcss, test_library_crit9_media_list.py.
<!-- SECTION:NOTES:END -->
