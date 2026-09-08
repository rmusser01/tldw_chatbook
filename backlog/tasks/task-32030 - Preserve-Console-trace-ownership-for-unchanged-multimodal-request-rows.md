---
id: TASK-32030
title: Preserve Console trace ownership for unchanged multimodal request rows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:00'
updated_date: '2026-09-08 05:12'
labels:
  - console
  - bug
  - tracing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identical multimodal messages lose saved trace ownership when immutable containers are compared directly to provider payload containers, causing trace reservation to reject a send before generation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unchanged multimodal active user row retains its authoritative saved revision through trace request construction.
- [x] #2 Semantically changed provider rows remain artifacts and cannot acquire saved ownership by content similarity.
- [x] #3 The real trace boundary accepts an unchanged multimodal send while existing ownership and refusal tests remain green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression that sends unchanged multimodal JSON through the production agent request factory and SQLite trace boundary; add changed-text/image negative controls.
2. Normalize the immutable and mutable JSON container representations before exact descriptor comparison, retaining ordered matching and the existing saved-owner gates.
3. Run focused request/trace and existing controller ownership tests, lint and format changed ranges, self-review, and record evidence and limitations.
ADR required: no new ADR.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Correct JSON representation equality under the existing exact semantic ownership contract; no new schema, disclosure, ownership, or dispatch policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Normalized each incoming agent JSON row with the existing freeze_json helper before the ordered exact comparison against the frozen admitted request. Unchanged nested image content now retains its saved revision; changed text, image data, and content order remain artifacts and fail before reservation. No trace ownership, schema, disclosure, or dispatch policy changed; existing ADR-097 applies.
Added a real SQLite, production request-factory and gateway regression in test_console_trace_runtime.py. RED: the unchanged-image case failed at trace reservation while all three changed-content controls passed. GREEN: all four pass and the unchanged row produces one completed call owned by the current saved user. Whole targeted runtime/provenance and production two-turn/tool controller tests: 122 passed in 52.69 seconds. Final focused rerun: 4 passed in 2.47 seconds. Ruff reports zero introduced diagnostics against 23 inherited findings; changed ranges pass formatter checks and git diff --check passes. Self-review completed. Existing requests dependency warning and patch-tool docstring SyntaxWarning are unchanged. No full suite or real provider/profile was used.
Files: console_agent_bridge.py, test_console_trace_runtime.py, and an incident-based testing lesson. Work is isolated at origin/dev 3cccd9326 because the older shared checkout lacks the semantic trace architecture. The supplied reporter log does not identify whether its trigger was an image message, so reporter-specific resolution remains unconfirmed. A separately reproduced dictionary transform failure is tracked as TASK-32032.
<!-- SECTION:NOTES:END -->
