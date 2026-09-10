---
id: TASK-31998
title: Capture a coherent final inventory with optional external content
status: In Progress
assignee: []
created_date: 2026-09-07 23:56
labels:
- backup-recovery
dependencies:
- task-31978
- task-31993
- task-31994
- task-31995
- task-31996
- task-31997
updated_date: 2026-09-10 22:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: A completed capture reflects the final inventory under maintenance, including valid in-scope growth and coherent DB/asset dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed capture reflects the final inventory under maintenance, including valid in-scope growth and coherent DB/asset dependencies.
- [ ] #2 Changed scope or budget renews preview safely; partial and optional coverage is accurately reported.
- [ ] #3 Ordinary writers resume after verified capture, before encryption or output transfer.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement existing ADR126/component03 Task15: add immutable CaptureResult and compare_scope contract with mapping-change/growth tests; preflight options/output/capacity; obtain actual participant/native maintenance and rediscover scope under closed admission; capture/validate dependency groups and staged credential policy; bounded stable external files and truthful optional coverage; per-volume capacity/runtime cancellation/ENOSPC; resume ordinary writers before returning capture for packaging. Finish focused concurrency/scope/alias/tombstone/capacity evidence, scoped static review and task-owned commit. Runtime composition and credential work remain dependencies; no complete-capture claim before they qualify.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-15)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Public native capture reviewed with actual recovered-media payload and explicit external folder/empty-directory publication. Fixed external.files missing installed capture policy and SQLite preview mode=ro live WAL/SHM mutation: preview reads bounded private main+WAL copies, checks source stability/resources and retires actual connections before cleanup. Root capture/admission/public/staged-credential/writer cohort68 passed; exact native encrypted capture and actual age writer roundtrip2 passed after supplying existing offline Go caches and fake keyring fixture (no real credential test dependency). Reports /private/tmp/chatbook-capture-publication-review.md and /private/tmp/chatbook-preview-copy-report.md. New capture/writer modules full Ruff+format clean, touched root capture/storage Bandit0. Final credential shared-alias integration remains under review; owner/runtime gaps remain explicit, no Complete capability exposed.
Final focused integration: 146 passed in 28.22s across credentials, capture, native admission, public capture service, staged credentials, archive writer and RAG discovery/indexing. Includes real encrypted helper roundtrip, recovered-media payload, external files/empty directories and source-preserving SQLite preview. /private/tmp/chatbook-final-capture-integration.log. Owner census 11 passed in 10.88s after explicit scratch/control rows, without promoting pending owners. Fatal Ruff and diff checks clean; touched production Bandit has only the same 7 existing private_sqlite findings. Audio-history approval is pending after automatic review rejection; no Complete capability or task completion claimed.
Read-only delivery check during RAG public qualification identified a concrete remaining inventory issue: real admission_authority creates default-config-parent/recovery-bootstrap, and inventory._unknown_children(..., 'shared_config') then labels that installed fixed control directory unknown. RAG fixtures intentionally remain partial, so passing owner tests do not qualify a complete profile. Before complete-capture promotion, classify fixed bootstrap and locally registered recovery/rollback/staging control roots under the existing explicit exclusion policy (spec default exclusions and independent local activation authority), without broad name-based exclusions or weakening unknown-file detection. Investigate and regression-test this narrow existing control-root classification in a separate original Task15 increment; no production change yet.
Fixed installed recovery-control inventory unit reviewed and qualified: exact default bootstrap root is intentionally excluded only after strict private control/registry/marker validation; malformed authority is unavailable, absent authority creates nothing, and unknown lookalikes remain unknown. Public capture retains distinct discovery/session exclusion IDs. Root independent code review approved scoped two-file change. Root reran test_recovery_control_inventory.py, test_capture.py, test_capture_service.py: 39 passed (14.89s). Worker broader cohort69passed/2fail; both failures independently reproduced with pre-edit inventory: test_normal_config_and_pure_resolvers_agree (raw_source_selection_changed before discovery), test_fresh_process_discovery_imports_no_bootstrap_or_services_and_changes_nothing (existing recovered_media DB import). These remain recorded, not suppressed. Production census delta0, Bandit0→0, Ruff existing I001 unchanged; isolated report /private/tmp/chatbook-control-inventory-report.md. Task15 remains In Progress for complete inventory coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->