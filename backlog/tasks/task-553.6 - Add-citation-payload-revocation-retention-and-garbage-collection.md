---
id: TASK-553.6
title: Add citation payload revocation retention and garbage collection
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 11:07'
labels:
  - rag
  - citations
  - privacy
  - retention
dependencies:
  - TASK-553.5
  - TASK-553.3
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enforce revocation-scoped deduplication, durable non-content tombstones, and reference-safe payload collection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Snapshot deduplication includes tenant or profile, authority, confidentiality policy, revocation scope, and governed exact-content identity.
- [x] #2 Revocation and secure purge clear governed run, snapshot, and answer-attempt fields while retaining sealed run, attempt, evidence-reference, and payload identities plus only the permitted non-content tombstone.
- [x] #3 Cache, import, and Sync replay cannot resurrect a tombstoned origin payload.
- [x] #4 Garbage collection respects message and artifact owners, pending links, soft-delete retention, Sync tombstones, and policy retention windows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing revocation-scoped dedupe, secure purge, tombstone, anti-resurrection, and identity-preservation tests.
2. Implement strict frozen lifecycle policy/tombstone contracts and atomic revoke transactions that clear every governed run/snapshot/attempt field without rewriting sealed trace structure.
3. Enforce tombstone checks on repository write, cache, imported, simulated Sync replay, and hydration seams so purged origins cannot resurrect.
4. Add reference-safe GC with message/artifact/pending-operation/Sync/policy-retention barriers and topologically safe graph deletion.
5. Run focused lifecycle/repository plus adjacent identity/persistence/migration/benchmark regressions, lint, diff, and both independent review gates.
6. Complete acceptance criteria and implementation notes only after approval.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This task implements ADR-024’s accepted revocation, tombstone, retention, anti-resurrection, and owner-safe collection policy using the fixed v25 schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict lifecycle contracts for five-part revocation-scoped snapshot dedupe, retention, durable non-content tombstones, atomic revoke, and bounded reference-safe collection. Dedupe requires non-null governance scope, authority, confidentiality policy, revocation scope, and secret-scoped exact-content identity; cross-profile/tenant/authority/policy/scope values remain isolated.

Revocation validates every snapshot sharing the tombstone origin key before writes, rejects governance/policy collisions, inserts one scope-bound tombstone, and clears every governed snapshot field plus run and selected/diagnostic answer payloads for all referencing traces while preserving sealed aggregates, completeness, refs, markers, and opaque identities. Repository write/retry/cache/import/Sync-simulated/hydration and active-presentation seams honor tombstones. Complete-at-seal traces refresh to a verified ACTIVE presentation with only evidence_revoked warning while hydration remains REVOKED/no governed data; non-complete seals and corrupt/purged rows without an exact origin+revocation-scope tombstone remain nonactive.

GC respects active and retained soft-deleted message owners, nonreleased artifact leases, pending/applied operations, caller Sync barriers, and policy windows. It deletes graphs topologically and only removes unreferenced snapshots/tombstones. A profile-bound opaque cursor carries independent trace/tombstone keyset positions, bounded 32-row minimum pages, high-water fairness, deterministic resume/wrap, and final transactional barrier rechecks. Tombstones cannot expire while any surviving same-origin snapshot/ref needs anti-resurrection. Writer locking touches only the singleton identity row.

Verification: 414 full citation/persistence/migration/performance tests passed; final focused quality review passed 139 tests; qualification at 30 samples/5 warmups had overall_pass=true; Ruff check/format and git diff checks passed. Independent specification and quality/security reviews approved the final implementation with no remaining Critical or Important findings.

ADR required: yes. Applied existing backlog/decisions/024-rag-citation-provenance-and-source-resolution.md; no new ADR was needed.
<!-- SECTION:NOTES:END -->
