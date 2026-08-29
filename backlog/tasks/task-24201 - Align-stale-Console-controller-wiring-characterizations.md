---
id: TASK-24201
title: Align stale Console controller wiring characterizations
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 13:11'
updated_date: '2026-08-29 13:13'
labels:
  - console
  - tests
  - concurrency
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair two Console wiring tests that still assert pre-decomposition dependency seams and pre-worker-group call arguments, so the containing wiring suite validates the current controller graph.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review-selection native-message characterization patches the current ConsoleMessageController seam
- [x] #2 Realtime worker characterization supplies and expects its explicit work group
- [x] #3 The complete Console controller wiring test file passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only alignment with existing Console decomposition and explicit worker-group contracts; no production behavior change. Update the review-selection test to replace ConsoleMessageController._native_console_messages, update the realtime test to pass and expect a named group, run both exact nodes then the complete wiring file, static checks, record evidence, and close.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned two stale test seams with already-shipped Console ownership. The review-selection characterization now replaces ConsoleMessageController._native_console_messages, matching wiring.py's current cross-controller late binding. The realtime characterization now supplies a console-realtime-test group and verifies it is forwarded, matching the existing adapter's explicit-group contract. No production code changed. ADR required: no; ADR path: N/A. Verification: exact two regressions 2 passed in 2.32s; complete Console controller wiring file 30 passed in 7.15s; Ruff check passed; Ruff format check passed; compileall passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
