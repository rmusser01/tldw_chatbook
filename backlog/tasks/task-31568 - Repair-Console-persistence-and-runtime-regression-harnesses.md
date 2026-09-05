---
id: TASK-31568
title: Repair Console persistence and runtime regression harnesses
status: Done
assignee: []
created_date: '2026-09-05 04:09'
updated_date: '2026-09-05 04:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore failing Console regression coverage whose test harnesses no longer satisfy current workspace persistence and runtime construction contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced Console persistence and runtime harness failures pass
- [x] #2 Affected Console test modules pass in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the current annotation, native-chat, and runtime-double failures. 2. Update test fixtures only where production ownership and construction contracts are intentional; fix production only when real behavior is defective. 3. Run focused regressions and each affected module in full. ADR required: no. ADR path: N/A. Reason: this is regression harness maintenance for existing Console boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated Console test harnesses to use the active app workspace registry, current
prompt-improvement gateway signature, real chat-host construction, owner-thread
shutdown semantics, and current runtime UI/state contracts. The affected annotation,
native-chat, and MCP approval modules pass in full: 462 tests passed. Ruff and diff
validation were run on the changed files. ADR required: no; the changes preserve
existing Console boundaries and introduce no architectural decision.
<!-- SECTION:NOTES:END -->
