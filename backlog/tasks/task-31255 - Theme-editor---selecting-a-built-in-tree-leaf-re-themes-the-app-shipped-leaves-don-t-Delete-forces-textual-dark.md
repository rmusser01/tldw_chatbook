---
id: TASK-31255
title: Theme editor - selecting a built-in tree leaf re-themes the app, shipped leaves
  don't, Delete forces textual-dark
status: To Do
created_date: 2026-09-04 05:24
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
load_theme sets self.app.theme immediately when the leaf is textual-dark or textual-light but not for the 58 shipped themes, so browsing the tree changes the running app inconsistently and there is no undo. _delete_user_theme ends with load_theme('textual-dark'), which re-themes the whole app after a dialog that only promised to remove a file. Live: textual-dark leaf re-themed the app; agentic_terminal did not. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting any tree leaf loads it for editing without changing the running app's theme; Apply is the only editor action that changes app.theme
- [ ] #2 After Delete the editor reloads the previously selected theme (or the first available one) and app.theme is unchanged
- [ ] #3 The existing delete tests are updated to assert the reloaded editor state instead of a forced textual-dark app theme
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
