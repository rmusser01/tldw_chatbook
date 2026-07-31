---
id: TASK-1613
title: >-
  Pin the zero-config create-target error path
status: To Do
assignee: []
created_date: '2026-07-31 15:10'
labels:
  - evals
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-1482 review coverage gap. Clicking "Create target from configured llama.cpp server" with no llama_cpp URL configured notifies "No llama.cpp server is configured; set one in Settings first." — verified correct by a reviewer's live probe, but no test pins the copy or the no-row-created outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A test drives the zero-config click and asserts the exact toast and that no eval_models row is created
<!-- AC:END -->
