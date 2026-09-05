---
id: TASK-31251
title: Theme editor - Name box is write-only, so Apply, Export, Reset and Delete act
  on a stale name
status: Done
created_date: 2026-09-04 05:23
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: high
updated_date: 2026-09-04 06:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
#settings-theme-name has no Input.Changed handler. Save reads the box, but Apply (toast + registered name), Export (file name), Reset (file lookup) and Delete (file lookup) read self.current_theme_name, which only changes on tree selection, New or Clone. Live: New -> rename to 'ocean' -> Apply says "Theme 'new_theme' applied"; Save writes ocean.toml; Reset reverts the Name box to new_theme and reports success; Delete says "No saved custom theme named 'new_theme'". Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Editing the Name box updates the name used by Apply, Export, Reset and Delete without pressing Save
- [x] #2 After Save, Reset reloads the saved file under the saved name and the Name box keeps that name
- [x] #3 Apply's toast and the registered runtime theme use the name currently shown in the Name box
- [x] #4 A regression test walks New -> rename -> Apply -> Save -> Reset -> Delete and asserts the name at each step
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added @on(Input.Changed, '#settings-theme-name') keeping current_theme_name in step with the box; Reset of a renamed never-saved theme now warns 'No saved version ... to reset to' instead of reverting the box and claiming success. Regression test walks New -> rename -> Apply -> Save -> Reset -> Delete.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
