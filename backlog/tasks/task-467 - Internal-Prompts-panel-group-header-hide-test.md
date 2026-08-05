---
id: TASK-467
title: 'Internal Prompts panel: test that a subsystem header hides when search filters all its rows'
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
InternalPromptsPanel._on_search hides a subsystem's group header when all its rows are filtered out (any_visible flag). The behavior is correct by inspection but has no direct test. Add one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test searches for a term matching only one subsystem and asserts the non-matching subsystems' group headers have display=False
- [ ] #2 The matching subsystem's header remains visible
<!-- AC:END -->
