---
id: TASK-546
title: Repair legacy conversation migration fixtures for v21 world-book invariant
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 20:22'
updated_date: '2026-07-24 20:22'
labels:
  - database
  - migrations
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make legacy v12/v13 conversation migration fixtures represent the historical schema so current migrations are tested without weakening fail-closed production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy v12 and v13 fixture databases include the world-book tables that existed from schema v9 without the v21 priority column
- [x] #2 Conversation parity migrations reach the current schema and preserve asserted rows
- [ ] #3 The focused migration and full-suite fail-fast gates pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: This is a test-fixture correction that preserves the existing
migration boundary and ADRs.

1. Add a regression asserting the legacy fixture contains the historical
   pre-v21 world-book shape.
2. Extend the shared v12/v13 fixture with the v9 world-book tables and no
   priority column.
3. Run focused migration/parity tests, the diagnostic sentinel, and resume the
   full-suite fail-fast gate.
<!-- SECTION:PLAN:END -->
