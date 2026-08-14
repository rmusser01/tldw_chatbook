---
id: TASK-16302
title: Repair stale ingest worker test contract
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-14 01:53'
labels:
  - tests
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the full-suite baseline by aligning one App ingest test with the current parse-worker request identity contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The invalid-audio routing test accepts and verifies the current three-argument parse-worker call
- [ ] #2 The complete test node and App test module pass without production changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the failing seven-case baseline and current production call contract.
2. Update only the stale test unpack and assert the exact generation/job identity.
3. Run the focused node, App module, Ruff, diff/privacy checks, and full-suite fail-fast gate.

ADR required: no
ADR path: N/A
Reason: This is a test-only repair to an existing worker call contract; no architecture or runtime policy changes.
<!-- SECTION:PLAN:END -->
