---
id: TASK-24197
title: Align skill-file runtime test with durable tool-content omission
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 12:28'
updated_date: '2026-08-29 12:32'
labels:
  - agents
  - tests
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the authorized skill_file runtime test after run-log privacy hardening intentionally stopped persisting raw tool content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Authorized skill_file content reaches the next model turn while durable run history records only the sanitized success receipt
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test maintenance for the existing TASK-24193 run-log privacy boundary; no runtime or persistence policy change. Replace the stale durable-content assertion with an ephemeral next-model-turn assertion plus an explicit sanitized durable receipt assertion, run the exact node and containing file, then static checks and close.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the full skill_file runtime module with the established TASK-24193 privacy boundary. Success content and refusal/error details are now asserted in the next live provider turn, while every durable tool_result is asserted as an outcome-only receipt with omitted result content. Added shared receipt helpers and covered authorized reads, unauthorized names, malformed readers, empty/absent bindings, and the real LocalSkillsService read seam. ADR required: no; ADR path: N/A; this changes test expectations only. Verification: exact regression 1 passed; containing module 8 passed in 0.48s; Ruff check passed; Ruff format check passed after formatting; compileall passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
