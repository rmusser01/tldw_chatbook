---
id: TASK-33071
title: Delete confirmation states the active/launch-default consequence
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. Deleting the active or launch-default theme only explains the fallback to Textual Dark in the toast after deletion. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Delete confirmation names the consequence when the theme is active and/or the launch default
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compute the delete consequence in request_delete from current_launch_default() and app.theme, mirroring _fall_back_after_delete's three branches.
2. Append it to the confirmation message; test each branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
_delete_consequence mirrors _fall_back_after_delete's three branches (active+launch -> switch to Textual Dark and launch with it; launch only -> launch with Textual Dark; active only -> switch to the launch theme, or Textual Dark when it is missing) and is appended to the Delete confirmation for readable themes. Tests assert each branch's line in test_settings_theme_file_api.py.
<!-- SECTION:NOTES:END -->
