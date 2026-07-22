---
id: TASK-468
title: 'Internal Prompts editor: debug-log the modal's unknown-action branch'
status: To Do
assignee: []
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - polish
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
InternalPromptsPanel._apply_editor_result has a silent `else: return` for a result action other than save/reset/None. It is currently unreachable (the modal's dismiss contract is closed), but a future modal change could hit it silently. Add a one-line debug log for defense-in-depth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unknown-action branch logs at debug level with the unexpected action value
- [ ] #2 No behavior change for the save/reset/None paths
<!-- AC:END -->
