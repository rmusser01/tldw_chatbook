---
id: TASK-31257
title: Theme editor - 'New starts a theme from the current palette' but New loads
  a hardcoded blue palette
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
The tree hint (settings-theme-tree-hint) says New starts from the current palette; _new_theme replaces the palette with a fixed blue set and dark=True. Live: textual-dark's #004578 became #0099FF. Copying the loaded palette matches the promise and Clone semantics. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New copies the currently loaded colours and dark flag into a theme named new_theme with the Name box focused
- [x] #2 test_theme_tree_has_empty_state_guidance and test_settings_theme_editor_new_confirms_before_discarding_edits still pass
- [x] #3 Docs/User_Guide/settings.md Theme section describes New as starting from the current palette
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
_new_theme copies the loaded palette and keeps the dark flag; the hardcoded blue set is only the fallback for an empty editor. Docs updated. Existing New/hint tests unchanged and green.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
