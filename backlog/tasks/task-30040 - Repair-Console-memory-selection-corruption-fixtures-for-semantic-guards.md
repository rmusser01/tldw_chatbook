---
id: TASK-30040
title: Repair Console memory-selection corruption fixtures for semantic guards
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 04:03'
updated_date: '2026-09-03 04:04'
labels:
  - console
  - testing
  - bug
dependencies: []
references:
  - backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Console memory-selection regression suite on current dev by keeping deliberate persisted-row corruption fixtures compatible with the fail-closed semantic mutation boundary, without weakening production guards or changing Console behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All current-dev failures in Tests/Chat/test_console_memory_selection.py caused by direct guarded fixture mutations pass.
- [ ] #2 Deliberate fixture corruption is authorized only for the exact message and message_update operation inside a managed transaction.
- [ ] #3 Production semantic mutation guards and Console behavior are unchanged.
- [ ] #4 Focused tests, Ruff checks on touched Python paths, git diff --check, and backlog task-ID uniqueness pass.
- [ ] #5 ADR required: no; this is a test-fixture compatibility repair under existing ADR-097.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the four current-dev red failures and confirm they are stale direct fixture mutations blocked by ADR-097.
2. Add one test-local, transaction-scoped helper that authorizes only message_update for the exact message, then route only the four guarded corruption writes through it.
3. Run the affected pytest module, Ruff 0.15.22 checks, backlog ID uniqueness, and git diff --check; self-review and close the task.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture compatibility repair under existing ADR-097 and changes no production architecture or behavior.

Detailed plan: Docs/superpowers/plans/2026-09-03-task-30040-console-memory-selection-fixture-guards.md
<!-- SECTION:PLAN:END -->
