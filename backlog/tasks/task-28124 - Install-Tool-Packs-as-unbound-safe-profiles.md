---
id: TASK-28124
title: Install Tool Packs as unbound safe profiles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 04:36'
updated_date: '2026-09-02 05:31'
labels:
  - tool-packs
  - import
  - activation
  - security
dependencies:
  - TASK-27058
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Revalidate an approved Tool Pack import review and install exactly its reviewed destination as a new unbound runtime permission profile, with receipt-first durability, lifecycle serialization, and strict ambiguity reconciliation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compilation emits a complete tool_pack_imported runtime profile with fail-closed fallbacks, current destination definition hashes, revision 1, canonical policy digest, receipt linkage, compact counts, and the authoritative first-bind confirmation marker without unresolved identity details.
- [x] #2 Activation re-reads and revalidates the archive, mappings, destination inventory, strict store generation, destination availability, references, review expiry, profile/byte caps, and receipt capacity before any authority write.
- [x] #3 The commit order is durable receipt ownership, lifecycle coordinator, store fence, active/archived reference check, and install-if-absent; stale or failed commits never overwrite, suffix, bind, or mutate a workspace.
- [x] #4 Ambiguous authority outcomes reconcile from strict bytes: exact installed state succeeds idempotently, exact prior absence fails, and any third state returns activation_uncertain without retry.
- [x] #5 Import alone leaves existing workspaces and effective policies unchanged, the installed profile has zero references and requires first-bind confirmation, and targeted activation/resolver tests plus scoped static checks pass.
- [x] #6 Exact automatic matches activate without fabricating a manual server mapping; detailed import receipts accept and canonically round-trip an empty reviewed-mapping list while preserving all existing mapping validation and bounds.
- [x] #7 Detailed receipts faithfully preserve Task 8's overlapping diagnostic/action categories for the same exact identity, while rejecting case-fold aliases, matched/unmatched overlap, changed/missing overlap, pending-Deny/omitted overlap, and more than 2,000 distinct identities.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing receipt regressions, then correct the contradictory Task 5 admission rules so canonical empty mappings and Task 8's constrained exact-identity category overlaps round-trip; keep category-conflict, case-fold alias, and distinct-identity bounds strict, then verify the receipt-store suite.
2. Add failing safe-compilation and lifecycle-sentinel tests for fallbacks, destination hashes, imported metadata, and privacy.
3. Implement exact profile compilation from an immutable review and a fresh complete destination inventory.
4. Add failing stale-review, receipt ordering, install-if-absent, ambiguous reconciliation, and unbound-workspace integration tests.
5. Implement receipt-first activation under lifecycle coordinator and store fence with strict post-exception reconciliation.
6. Run focused activation/resolver and related receipt/contract/import tests, scoped Ruff, diff hygiene, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes receipt-first unbound activation, exact/manual revalidation, install-if-absent authority semantics, first-bind lifecycle state, and strict outcome reconciliation.

Plan deviation: Task 9's RED cycles exposed that the Task 5 receipt parser rejected
both `reviewed_mappings=[]` and every cross-group identity repetition, making the
ADR-required exact-match/no-manual-mapping import and Task 8's intentionally
overlapping changed/missing plus pending-Deny/omitted classifications impossible.
The scoped file set therefore also modifies
`tldw_chatbook/Tool_Packs/receipt_store.py` and
`Tests/Tool_Packs/test_receipt_store.py` to accept canonical empty mappings and the
same exact identity in one compatible diagnostic/action pair. Ordering, per-group
uniqueness, case-fold alias rejection, incompatible-category rejection, privacy,
and the 2,000-distinct-identity capacity bound remain strict.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a pure compiler for schema-1 `tool_pack_imported` profiles with fail-closed MCP-global, builtin, and per-server fallbacks, current destination definition hashes, compact lifecycle counts, revision 1, canonical policy digest, durable receipt linkage, and the authoritative first-bind marker.
- Added receipt-first unbound activation that re-inspects the exact archive/id/mappings, captures fresh inventory, durably writes and live-owns the receipt, then enters lifecycle coordinator → store fence for final expiry/store/id/reference validation and install-if-absent.
- Added strict ambiguity reconciliation: exact installed profile bytes succeed without retry, absence returns a stable failure, and any third/unreadable state returns `activation_uncertain`. Existing workspace defaults and effective policy remain unchanged and the installed profile has zero references.
- Resolved two prerequisite contract contradictions found by RED tests: canonical empty reviewed mappings now round-trip, and receipts preserve Task 8's compatible diagnostic/action overlaps while rejecting aliases, contradictory overlaps, and more than 2,000 distinct identities.
- Fresh controller verification passed 85 activation/resolver tests and a 310-test receipt/contract/catalog/import/authority matrix. All five touched files pass scoped Ruff lint/format and diff hygiene; the only warning is the known environment-level Requests dependency warning. Independent review found no Critical, Important, or Minor issue and passed 193 focused tests.
- ADR check: no new ADR required; implementation follows `backlog/decisions/107-portable-tool-use-packs.md`. No schema migration, dependency, automatic binding, workspace mutation, overwrite, or id suffixing was added.
<!-- SECTION:NOTES:END -->
