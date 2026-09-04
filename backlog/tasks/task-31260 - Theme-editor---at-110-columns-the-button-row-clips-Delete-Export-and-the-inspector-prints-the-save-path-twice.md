---
id: TASK-31260
title: Theme editor - at 110 columns the button row clips Delete/Export and the inspector
  prints the save path twice
status: Done
created_date: 2026-09-04 05:24
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: medium
updated_date: 2026-09-04 06:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With the middle pane at ~45 cells, the four 16-cell buttons in the library row overflow (Delete and Export are cut off), the fixed 24-cell label column eats half the width, and the Scope Inspector prints the absolute themes directory twice, wrapped over five lines each. Observed live at 110x36. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 110 columns all library and action buttons are visible and clickable (rows wrap or stack under the settings compact mode)
- [x] #2 The inspector shows the themes directory once, abbreviated with ~ for the home directory
- [x] #3 A geometry test at 110x36 asserts every editor button is inside the card region
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Compact workbench (<=100 cols) stacks the editor's action rows and gives each button full width, keyed to a theme-editor-action class (fast-path ratchet stays 274). Inspector shows the themes directory once via _display_path (~ for home). Render test at 110x36 asserts every button is inside the pane; unit test for _display_path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
