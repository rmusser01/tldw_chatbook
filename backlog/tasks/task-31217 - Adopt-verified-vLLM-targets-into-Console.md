---
id: TASK-31217
title: Adopt verified vLLM targets into Console
status: To Do
assignee: []
created_date: '2026-09-03 22:33'
labels:
  - vllm
  - lab
  - console
  - handoff
dependencies:
  - TASK-31215
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the Lab workflow by applying a verified vLLM provider, canonical endpoint, and served model to Console with explicit session or durable scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Use in Console is enabled only for the current verified vLLM generation.
- [ ] #2 Session adoption updates the active Console provider, endpoint, model, and readiness without writing durable configuration.
- [ ] #3 The durable option delegates to the established Settings/provider persistence path and never silently replaces a different configured endpoint.
- [ ] #4 Wildcard bind addresses are converted to an explicit usable client endpoint without weakening exposure warnings.
- [ ] #5 Mounted Lab-to-Console and persistence regression tests cover session, durable, stale, and rollback paths.
<!-- AC:END -->
