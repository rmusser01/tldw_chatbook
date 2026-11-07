---
id: TASK-34990
title: Add reversible conversation archive lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:33'
updated_date: '2026-09-10 16:36'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Saved conversations need reversible archive state without losing history or identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Individual and bulk archive preserve conversation identity and messages with reversible version-checked changes.
- [x] #2 Active Archived and All queries are correctly filtered before pagination and counts.
- [x] #3 Targeted persistence and migration tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/144-conversation-archive-and-exact-resume.md
Reason: durable lifecycle and query contract.

1. Read spec and existing migration/service patterns.
2. Write failing real SQLite lifecycle/pagination/migration tests.
3. Implement archive persistence, version-checked bulk mutation, archive scope and metadata projection.
4. Run targeted tests and lint; document evidence.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented schema 43 local archive flag and index under ADR-144, version-checked batch archive/restore receipts and 500-ID archive-state reads. Local default-active archive_scope now filters list/count/search/locator before pagination; exact-ID readers and whole-library export retain archived history. Chat scope delegates archive/state/reader I/O off the event loop except per-thread in-memory SQLite. Real SQLite coverage verifies restart, identity/messages/workflow retention, deleted-state independence, partial/stale/no-op receipts, pagination/locator/search, migration rollback, and competing writers. Initial new tests failed before implementation. Focused run: 207 passed; six repeated competing-writer checks passed. Ruff new tests and fatal-error checks pass; production full-rule counts unchanged from baseline (500 DB, 5 conversation service, 4 scope service). Two migration tests asserting schema 40 fail against the original schema-42 baseline as well; final broader targeted run excludes those known failures. Integration/status approval remains with parent task; no commit made.

Final integrated evidence: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. User guide updated at Docs/User_Guide/console/sessions-tabs-workspaces.md. ADR-144 linked. Compact/wide mounted flows and final cancellation-safe lifecycle run pass (29 checks); targeted FTS/recovery run passes (75); six close tests pass. Independent final review reports no remaining blockers. No full suite, live provider calls, commit or unrelated checkout cleanup.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from task-34951 to TASK-34990 (2026-10-14): the id collided
with a task already on dev; per the 2026-08-21 owner rule (TASK-19601)
the dev-side arrival keeps the id on merge.
