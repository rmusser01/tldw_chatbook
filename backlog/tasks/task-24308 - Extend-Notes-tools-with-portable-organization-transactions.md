---
id: TASK-24308
title: Extend Notes tools with portable organization transactions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 23:44'
labels:
  - notes
  - agents
  - tools
dependencies:
  - TASK-24307
documentation:
  - >-
    Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - >-
    Docs/superpowers/plans/2026-08-29-notes-organization-agent-tool-transactions.md
  - backlog/decisions/105-portable-notes-organization-and-agent-lessons.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give permitted agents exact folder and keyword discovery plus conflict-safe, additive organization-aware note saves, including durable local pending states when portable organization cannot yet publish.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `library_search_notes` supports spelling-exact keyword and unambiguous exact folder filters while retaining bounded lexical search and pagination
- [x] #2 Search and get responses expose bounded folder and keyword metadata plus stable public identities and the current `organization_version`
- [x] #3 `library_save_note` can add requested keywords without removing user keywords and rejects stale note or organization state without overwriting concurrent user changes
- [x] #4 Note content, requested organization, and immutable local synchronization intents commit or roll back together inside the owning Notes database
- [x] #5 A permitted lesson save made before organization readiness remains locally discoverable and excluded from every normal dispatcher until atomic finalization
- [x] #6 Folder-only collisions survive restart as non-blocking placement review, while deletion or permission denial leaves no orphaned receipt or hidden write
- [x] #7 Targeted tool-parity, transaction-failure, restart, permission, pagination, and concurrency tests pass across Console and in-app MCP surfaces
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the v59 Notes-owned receipt schema using genuine v58 reopen coverage, then add scoped note publication intents through a genuine v59→v60 migration when finalization requires that new durable owner.
2. Extend exact Notes folder/keyword query filters and bounded portable organization metadata.
3. Extend the single shared Console/MCP public tool contract.
4. Implement atomic note plus additive organization saves with concurrency checks and pending receipts.
5. Route shared Library search/get/save through the Notes-owned service with stable public errors.
6. Finalize pending receipts atomically and exclude them from every normal dispatcher until ready.
7. Document, run targeted and schema-safe verification, self-review, and close TASK-24308.

ADR required: yes
ADR path: backlog/decisions/105-portable-notes-organization-and-agent-lessons.md
Reason: This task implements ADR-105 persistent receipts, concurrency tokens, transaction ownership, and dispatch boundaries; no new ADR is needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the [ADR-105](../decisions/105-portable-notes-organization-and-agent-lessons.md)
transaction boundary through the existing shared Console/MCP Library service.

- Added exact Notes keyword/folder selectors, bounded portable folder/keyword
  metadata, stable public identities, and organization concurrency tokens. The
  keyword filter is spelling-exact; folder paths use the portable case-fold-only
  resolver; selector resolution and paging share one Notes read snapshot.
- Extended the single descriptor contract and Notes-owned save so keyword and
  folder changes are additive, stale content/organization is refused, and all
  Notes mutations, organization intents, and receipts share one SQLite
  transaction. Policy denial occurs before backend access.
- Added content-free blocking and placement-review receipts in schema v57. A
  pending note remains locally discoverable and excluded from ordinary
  dispatchers; readiness, review resolution/dismissal, deletion, and restart
  preserve explicit atomic outcomes.
- Added Notes-owned scoped publication intents through a genuine v59→v60
  migration, including SQL allowlisting and index-census coverage. Intents are
  retained until general-outbox acknowledgement and drain in note/entity-version
  lineage order rather than wall-clock order.
- Added contract, parity, transaction-failure, migration/reopen, query-plan,
  restart, permission, collision, pagination, and concurrency coverage across
  Console and in-app MCP. The prescribed Task 7 matrix passed 541 tests with one
  existing `RequestsDependencyWarning`; compileall passed. Supplemental causal
  ordering, v58 migration, and focused schema-census gates passed 2, 4, and 4
  tests respectively.
- Full-repository tests were not run, per repository policy. The schema-safe
  live gate used disposable root `/tmp/tldw-task24308-live.Nb8MgD`: HOME,
  USERPROFILE, every XDG directory, TMPDIR, effective `TLDW_CONFIG_PATH`, and
  `[paths].data_dir` remained under that root; worktree provenance was asserted;
  the real app launched and exited through Ctrl+Q with status 0; and its isolated
  ChaChaNotes database reached schema v58 with the receipt, publication-intent,
  folder, and keyword tables. Real current-server transport UAT was unavailable
  because the isolated profile intentionally had no endpoint or credentials; no
  transport result is claimed.

Lessons earned by the real v57-reopen and reversed-clock dispatch incidents are
recorded in [lessons-testing-evidence.md](../docs/lessons-testing-evidence.md).
The parent closure pass found no competing TASK-24308 or ADR-105 claimant and
updated the task to **Done** through Backlog.md.
<!-- SECTION:NOTES:END -->
