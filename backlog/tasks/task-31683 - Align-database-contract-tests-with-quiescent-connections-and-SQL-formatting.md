---
id: TASK-31683
title: Align database contract tests with quiescent connections and SQL formatting
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:22'
updated_date: '2026-09-05 18:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep private SQLite and bounded resume projection assertions accurate after existing connection maintenance and SQL layout changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ChaChaNotes connection assertions pin the actual quiescent connection factory alongside unchanged private-file and PRAGMA contracts.
- [x] #2 Resume projection checks retain byte bounds and pre-JSON filtering without depending on whitespace inside the CAST expression.
- [x] #3 Both complete database files pass with scoped static checks and no production changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the exact3 failures and read existing compaction/connection ownership decision. 2. Pin the new existing _QuiescentSQLiteConnection factory in the ChaChaNotes kwargs expectation, preserving all filesystem and PRAGMA assertions. 3. Match whitespace-flexible assistant-message byte-bound SQL inside the materialized eligibility stage while retaining exact1..128 limit and all other bounds. 4. Run both complete database files and scoped static checks. ADR required: no. ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md (existing). Reason: test-only contract correction; no mutation authority, storage or connection behavior change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pinned the current quiescent SQLite factory installed by23113.11 while preserving all private-file, connection reuse and PRAGMA assertions. Made only the assistant-message CAST whitespace flexible and strengthened that bound to remain inside the materialized eligibility stage with exact1..128 bytes; all other pre-JSON bounds remain. Baseline3failed164passed; final167passed4.55s (/private/tmp/tldw-review-db-contracts-final-20260905.xml). Ruff, changed-range formatting, diff whitespace and self-review passed. Test-only, governed existing ADR097 connection maintenance; no new ADR.
<!-- SECTION:NOTES:END -->
