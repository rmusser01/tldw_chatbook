---
id: TASK-30040
title: Repair Console memory-selection corruption fixtures for semantic guards
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 04:03'
updated_date: '2026-09-03 04:13'
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
- [x] #1 All current-dev failures in Tests/Chat/test_console_memory_selection.py caused by direct guarded fixture mutations pass.
- [x] #2 Deliberate fixture corruption is authorized only for the exact message and message_update operation inside a managed transaction.
- [x] #3 Production semantic mutation guards and Console behavior are unchanged.
- [x] #4 Focused tests, Ruff checks on touched Python paths, git diff --check, and backlog task-ID uniqueness pass.
- [x] #5 ADR required: no; this is a test-fixture compatibility repair under existing ADR-097.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the current-dev Console memory-selection suite without changing production behavior or weakening ADR-097. Added one test-local helper that uses the exact database connection's coordinator capability inside a managed transaction and authorizes only message_update for the supplied message ID; only the four deliberate guarded corruption writes use it.

TDD evidence: before the repair, /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_memory_selection.py reported 4 failed and 32 passed because the guarded raw updates raised semantic mutation authorization required. After the repair and formatting, the same command reported 36 passed with one dependency warning; the successful process also emitted unrelated pre-existing pytest temporary-directory cleanup warnings. Ruff 0.15.22 check passed and format --check reported one file already formatted. The Python 3.12.11 AST/comment/directive guard matched before and after Ruff formatting. Tests/CI/test_backlog_task_id_uniqueness.py reported 3 passed with one warning, and git diff --check passed.

Files: Tests/Chat/test_console_memory_selection.py; Docs/superpowers/plans/2026-09-03-task-30040-console-memory-selection-fixture-guards.md; this task record. Independent review found no Critical or Minor issues and no implementation issue; its sole Important closeout finding is addressed by these notes and the completed checklist.

ADR required: no. Existing backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md governs the mutation boundary; this repair changes only test-fixture setup.
<!-- SECTION:NOTES:END -->
