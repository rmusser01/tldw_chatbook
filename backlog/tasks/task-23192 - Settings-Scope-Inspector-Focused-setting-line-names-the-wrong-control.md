---
id: TASK-23192
title: Settings Scope Inspector 'Focused setting' line names the wrong control
status: To Do
assignee: []
created_date: '2026-08-29 02:25'
labels:
  - ux
  - settings
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Scope Inspector's 'Focused setting' line read 'Appearance defaults' while the Reduce motion control demonstrably held focus (verified by a style-diff showing the focused background on that control). The line exists to tell keyboard users what their focus is currently on, which is exactly the guarantee TASK-23109's setting-level landing depends on -- a wrong name is worse than no line, because it contradicts what the user can see. Observed during the TASK-23109 verification pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 'Focused setting' line names the control that actually holds focus, including controls reached by search landing and by plain Tab traversal
- [ ] #2 When focus is on a container or a non-setting control, the line says so rather than naming an unrelated setting
- [ ] #3 A mounted test focuses at least two distinct settings and asserts the inspector line matches each
<!-- AC:END -->
