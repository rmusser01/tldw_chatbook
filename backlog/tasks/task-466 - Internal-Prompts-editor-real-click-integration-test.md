---
id: TASK-466
title: 'Internal Prompts editor: real-click integration test for Save/Reset/Cancel'
status: To Do
assignee: []
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - test-coverage
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The InternalPromptEditorModal's Save/Reset/Cancel are exercised via the `_save_from_test` seam and the panel's `_apply_editor_result` entry point, but no test drives the real Button.Pressed -> push_screen -> callback -> run_worker chain. A wrong `@on` selector (e.g. on Reset) would go undetected. Add a pilot-driven integration test that clicks the actual buttons.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pilot test clicks the modal's Save button and asserts the override persists + the row badge updates
- [ ] #2 A pilot test clicks Reset and asserts the override is cleared
- [ ] #3 A pilot test clicks Cancel (and presses Escape) and asserts no change
<!-- AC:END -->
