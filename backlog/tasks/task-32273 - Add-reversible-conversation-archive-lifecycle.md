---
id: TASK-32273
title: Add reversible conversation archive lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:33'
updated_date: '2026-09-10 19:56'
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
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: durable lifecycle and query contract.

1. Read spec and existing migration/service patterns.
2. Write failing real SQLite lifecycle/pagination/migration tests.
3. Implement archive persistence, version-checked bulk mutation, archive scope and metadata projection.
4. Run targeted tests and lint; document evidence.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md

PR review corrections (2026-09-10):
5. Prevent archive-only sync events while retaining optimistic receipt versions; execute the canonical v71 artifact and document ADR-147.
6. Bound exact-title lookup pages, strengthen archive-scope forwarding assertions, and adopt the shared scope validator.
7. Preserve the latest shared sync payload through archive-only version bumps in both trigger and maintenance retention.
8. Run targeted DB/service checks and self-review before restoring Done status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added indexed local archive state through schema migration 70→71, optimistic version-checked batch archive/restore, active/archived/all query filtering before counts and pagination, and bounded archive-state reads. Exact reads and exports retain archived history. Shared async mutations keep reservations through SQLite settlement and reject conflicting sends, drafts and live work.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.

PR #2576 backend review corrections: archive-only transitions no longer emit conversation sync updates, while local version increments continue to invalidate obsolete Undo receipts. Conversation retention now advances on emitted sync records, and its explicit sweep uses the latest emitted version, preserving the latest unsent shared payload through archive cycles. Shared payload edits (including mixed archive/content changes), deletion and undelete retain sync behavior. The v71 SQL artifact is now executed transactionally instead of duplicated inline. Exact-title reads expose validated limit/offset with default 100 and cap 1000, and deterministic ordering. Library delegate tests assert all three forwarded archive scopes; DB predicates use the shared strict scope validator. ADR-147 records the sync/version tradeoff.

Verification: 82 targeted DB archive/service tests passed. The retention follow-up passed 39 DB archive/retention cases; two raw hard-delete fixtures fail at the existing semantic-authorization guard, reproduced with unchanged HEAD DB code. Final focused verification passed 9 cases covering exact preservation through repeated archive cycles and explicit maintenance, removal of late obsolete payloads after archive/delete, mixed shared updates, migration rollback, soft-delete privacy, idempotence and writer census. SQL artifact execution, 1005-row title pagination/cap, stale Undo cycles, and all scope forwarding are covered. Ruff passed for modified tests, fatal production lint passed, and git diff --check passed. Self-review completed; no broad migration sweep was run.
<!-- SECTION:NOTES:END -->
