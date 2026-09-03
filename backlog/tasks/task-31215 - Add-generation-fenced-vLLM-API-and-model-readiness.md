---
id: TASK-31215
title: Add generation-fenced vLLM API and model readiness
status: To Do
assignee: []
created_date: '2026-09-03 22:32'
labels:
  - vllm
  - lab
  - readiness
dependencies:
  - TASK-31213
  - TASK-31214
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace process-liveness completion with an explicit, privacy-bounded vLLM lifecycle that proves the OpenAI-compatible API and served model are ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Lab distinguishes not configured, checking, launching, loading model, ready, stopping, and failed states.
- [ ] #2 Ready requires a current-generation bounded models-endpoint probe and an admissible exact served-model identity.
- [ ] #3 Cancellation, target edits, process death, recomposition, and newer checks prevent stale results from enabling actions.
- [ ] #4 Activity and recovery expose bounded categories without retaining credentials, raw commands, paths, or unrestricted child output outside the Lab-owned boundary.
- [ ] #5 Unit, loopback HTTP, lifecycle, privacy, and mounted UI tests cover the state machine.
<!-- AC:END -->
