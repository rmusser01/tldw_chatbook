---
id: TASK-28124
title: Install Tool Packs as unbound safe profiles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-02 04:36'
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
- [ ] #1 Compilation emits a complete tool_pack_imported runtime profile with fail-closed fallbacks, current destination definition hashes, revision 1, canonical policy digest, receipt linkage, compact counts, and the authoritative first-bind confirmation marker without unresolved identity details.
- [ ] #2 Activation re-reads and revalidates the archive, mappings, destination inventory, strict store generation, destination availability, references, review expiry, profile/byte caps, and receipt capacity before any authority write.
- [ ] #3 The commit order is durable receipt ownership, lifecycle coordinator, store fence, active/archived reference check, and install-if-absent; stale or failed commits never overwrite, suffix, bind, or mutate a workspace.
- [ ] #4 Ambiguous authority outcomes reconcile from strict bytes: exact installed state succeeds idempotently, exact prior absence fails, and any third state returns activation_uncertain without retry.
- [ ] #5 Import alone leaves existing workspaces and effective policies unchanged, the installed profile has zero references and requires first-bind confirmation, and targeted activation/resolver tests plus scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing safe-compilation and lifecycle-sentinel tests for fallbacks, destination hashes, imported metadata, and privacy.
2. Implement exact profile compilation from an immutable review and a fresh complete destination inventory.
3. Add failing stale-review, receipt ordering, install-if-absent, ambiguous reconciliation, and unbound-workspace integration tests.
4. Implement receipt-first activation under lifecycle coordinator and store fence with strict post-exception reconciliation.
5. Run focused activation/resolver and related contract/import tests, scoped Ruff, diff hygiene, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes receipt-first unbound activation, exact/manual revalidation, install-if-absent authority semantics, first-bind lifecycle state, and strict outcome reconciliation.
<!-- SECTION:PLAN:END -->
