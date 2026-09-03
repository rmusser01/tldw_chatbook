---
id: TASK-31219
title: Add current-server snapshots and reusable vLLM launch profiles
status: To Do
assignee: []
created_date: '2026-09-03 22:34'
labels:
  - vllm
  - lab
  - profiles
dependencies:
  - TASK-31215
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make repeated vLLM operation efficient and honest by separating the immutable running configuration from editable restart intent and retaining reusable non-secret launch profiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The UI displays an immutable current-server snapshot separately from the next-launch draft.
- [ ] #2 Edits made while running are labeled as next-restart changes and can be applied with one Restart with draft action.
- [ ] #3 Users can create, select, rename, duplicate, and delete named vLLM profiles containing only approved non-secret launch fields.
- [ ] #4 The last selected vLLM view and profile restore across screen recomposition and application restart.
- [ ] #5 Storage, migration if required, privacy, and profile lifecycle tests cover invalid, stale, and recovery states.
<!-- AC:END -->
