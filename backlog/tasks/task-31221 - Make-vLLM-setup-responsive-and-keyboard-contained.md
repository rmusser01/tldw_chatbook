---
id: TASK-31221
title: Make vLLM setup responsive and keyboard-contained
status: To Do
assignee: []
created_date: '2026-09-03 22:34'
labels:
  - vllm
  - lab
  - accessibility
  - responsive
dependencies:
  - TASK-31214
  - TASK-31215
  - TASK-31217
  - TASK-31219
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the complete vLLM setup, activity, profile, and Console-handoff workflow remains visible and operable at supported terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 80x24, 100x30, and 120x40 every visible focusable descendant remains within its owning pane.
- [ ] #2 Labels, inputs, and actions stack at compact widths without truncating the recovery or primary-action meaning.
- [ ] #3 Tab traversal stays within the active provider pane and lifecycle transitions move focus to the newly relevant action.
- [ ] #4 Provider navigation uses one documented key meaning that does not conflict with the Lab footer.
- [ ] #5 Production-stylesheet compositor and keyboard tests cover first-run, loading, ready, failure, current-versus-next, and handoff states.
<!-- AC:END -->
