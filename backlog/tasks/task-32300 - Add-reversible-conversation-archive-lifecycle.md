---
id: TASK-32300
title: Add reversible conversation archive lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:33'
updated_date: '2026-09-10 21:43'
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
PR #2576 wave 2: include archived conversations in import conflict and unique-name checks; document all public archive-scope arguments; verify focused import/service tests.
PR #2576 third review: preserve deleted conversation discovery independently of archive scope; prevent failed new mutations from exposing prior Undo; recheck durable state before existing-session Resume; serialize Unicode name checks with restore writes; align workspace-archive action copy and navigation contracts. Add focused regressions and verify affected integrations. ADR required: no new ADR; implements existing ADR147 lifecycle/recovery boundaries.
PR #2576 fourth review: centralize resume-ID validation; keep memory-backed registry enrichment on its owning thread; verify confirmed close retains real saved history; fence recovery publication by request/revision ownership; check durable state for both warm and cold resume paths; reuse async workspace restore for receipt Undo. Add targeted regressions. ADR required: no new ADR; implement ADR147 ownership and recovery rules.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added indexed local archive state through schema migration 70→71, optimistic version-checked batch archive/restore, active/archived/all query filtering before counts and pagination, and bounded archive-state reads. Exact reads and exports retain archived history. Shared async mutations keep reservations through SQLite settlement and reject conflicting sends, drafts and live work.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.

PR #2576 backend review corrections: archive-only transitions no longer emit conversation sync updates, while local version increments continue to invalidate obsolete Undo receipts. Conversation retention now advances on emitted sync records, and its explicit sweep uses the latest emitted version, preserving the latest unsent shared payload through archive cycles. Shared payload edits (including mixed archive/content changes), deletion and undelete retain sync behavior. The v71 SQL artifact is now executed transactionally instead of duplicated inline. Exact-title reads expose validated limit/offset with default 100 and cap 1000, and deterministic ordering. Library delegate tests assert all three forwarded archive scopes; DB predicates use the shared strict scope validator. ADR-147 records the sync/version tradeoff.

Verification: 82 targeted DB archive/service tests passed. The retention follow-up passed 39 DB archive/retention cases; two raw hard-delete fixtures fail at the existing semantic-authorization guard, reproduced with unchanged HEAD DB code. Final focused verification passed 9 cases covering exact preservation through repeated archive cycles and explicit maintenance, removal of late obsolete payloads after archive/delete, mixed shared updates, migration rollback, soft-delete privacy, idempotence and writer census. SQL artifact execution, 1005-row title pagination/cap, stale Undo cycles, and all scope forwarding are covered. Ruff passed for modified tests, fatal production lint passed, and git diff --check passed. Self-review completed; no broad migration sweep was run.

PR #2576 wave 2 backend corrections: importer conflict detection and generated-name probes explicitly include archived non-deleted conversations. Real SQLite import tests prove Skip preserves an archived title and Rename avoids both its base title and archived numbered suffix. All public Chat service methods carrying archive_scope now document active/archived/all, defaults, and pre-pagination filtering. Existing ADR-147 applies. Verification: new import cases and existing import/service neighbors passed; no new lint diagnostics or whitespace errors.
Final integrated review verification and baseline limits are recorded in Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. All modified archive flows pass their targeted tests; the unrelated compact Overview assertion reproduces with the prior Settings implementation. Started-write cancellation preserves storage completion publication. Existing ADR147 applies.
PR #2576 third review: fixed Trash archive independence, failed-mutation Undo ownership, durable existing-tab Resume checks, serialized Unicode restore names, workspace recovery labels, and archive navigation contracts. Targeted real SQLite, recovery and mounted checks pass; third-review evidence and temporary host-disk interruption are recorded in the QA report. ADR147 applies and documents deletion-oriented scope. Self-review and scoped static checks complete.
PR #2576 fourth review: centralized resume validation, preserved in-memory SQLite ownership, verified confirmed close retains stored history, fenced recovery against newer request revisions, checked both warm/cold lifecycle state, and routed Console receipt Undo through async restore with expected-record checks. Evidence: 99 focused tests, 2 supersession-boundary cases, confirmed-close SQLite test, and 14 mounted lifecycle cases pass; static and diagnostic checks pass. See fourth-review QA section; ADR147 applies.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered the unmerged archive lifecycle task from TASK-32273 to TASK-32300 after PR #2575 landed the reasoning-history task with the same ID. The landed task keeps its external references under the 2026-09-08 landed-keeps-ID clarification in lessons-backlog-hygiene.md. Its add commit is f043db12b7 (2026-09-10 19:28 UTC); the archive task entered this PR at 19:30 UTC. A fresh all-ref/object and live-worktree sweep found maximum 32299 before selecting 32300. The archive plan and ADR references move with this task; reasoning-history references remain unchanged.
