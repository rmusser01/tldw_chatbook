---
id: TASK-40726
title: Build frozen recovery fixtures in explicit SQLite transactions
status: In Progress
created_date: 2026-11-13 21:20
assignee:
- '@codex'
updated_date: 2026-11-13 21:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Frozen recovery fixtures issue hundreds of DDL statements outside an explicit transaction, causing a durable commit for each declaration. Under concurrent VM storage load, four lineage tests exceed their unchanged45-second child deadline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frozen fixture catalogs and seed rows commit together before production recovery validation begins.
- [x] #2 Existing lineage coverage, migration/rollback/security assertions, subprocess isolation and all deadlines remain unchanged.
- [ ] #3 Affected lineage and restore parameterizations pass focused checks and the loaded required CI lanes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-fixture construction optimization only; no application transaction or recovery boundary changes.

1. Confirm sqlite3 context manager does not begin a transaction for DDL; compare local failing-node baseline and fixture setup paths.
2. Add explicit BEGIN only to the three frozen-catalog fixture builders in Canvas and response-rule recovery tests, preserving their existing context-managed commit before validation.
3. Run affected parameterizations and catalog checks, independently review, then verify loaded normal PR CI without changing deadlines.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added explicit BEGIN before frozen catalog declarations in three fixture builders: Canvas lineage child, Canvas hybrid catalog, and response-rule historical catalog. Existing context managers commit fixture DDL and seed rows before production validation. No declarations, seeds, migration/rollback/security assertions, subprocess isolation, production code or deadlines changed; synchronous durability stays enabled. Recorded the incident lesson.

SQLite experiment confirmed CREATE leaves in_transaction false without BEGIN and true with BEGIN. Original exact version81 test passed locally in4.99s total (4.52s call); the loaded CI child had exceeded45s. After the change all76 affected parameterizations passed in71.41s, including all4 formerly failing children and all48 restore variants. Ruff and whitespace/backlog checks passed. Independent review verified fresh connection state, FTS5 transaction compatibility, duplicate-DDL error behavior and committed byte-comparison baselines; no blockers. Live loaded verification remains pending, so this task stays InProgress. No ADR required for fixture-only setup optimization.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
