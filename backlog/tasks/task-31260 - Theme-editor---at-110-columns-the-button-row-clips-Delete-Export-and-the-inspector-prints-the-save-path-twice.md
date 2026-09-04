---
id: TASK-31260
title: Theme editor - at 110 columns the button row clips Delete/Export and the inspector
  prints the save path twice
status: To Do
created_date: 2026-09-04 05:24
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With the middle pane at ~45 cells, the four 16-cell buttons in the library row overflow (Delete and Export are cut off), the fixed 24-cell label column eats half the width, and the Scope Inspector prints the absolute themes directory twice, wrapped over five lines each. Observed live at 110x36. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 110 columns all library and action buttons are visible and clickable (rows wrap or stack under the settings compact mode)
- [ ] #2 The inspector shows the themes directory once, abbreviated with ~ for the home directory
- [ ] #3 A geometry test at 110x36 asserts every editor button is inside the card region
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
