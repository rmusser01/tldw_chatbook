---
id: TASK-32000
title: Plan and stage both restore destinations with explicit dependency mapping
status: In Progress
assignee: []
created_date: 2026-09-07 23:57
labels:
- backup-recovery
dependencies:
- task-31978
- task-31995
- task-31996
- task-31997
- task-31999
updated_date: 2026-09-10 20:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Both destination modes produce immutable explicit mappings with no writes to live sources or targets during planning/staging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both destination modes produce immutable explicit mappings with no writes to live sources or targets during planning/staging.
- [ ] #2 Replacement preserves unknown data until reviewed and includes managed retirement/rollback scope.
- [ ] #3 Isolated restore works independently of damaged current config and rejects source aliases or untrusted destination authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task17 and ADR126 only. Validate immutable archive first; build explicit isolated/replacement restore/retire/preserve mappings from caller-supplied local destinations and independently discovered target inventory. Refuse partial replacement, unknown ownership, aliases/collisions/shared effects and changed target fingerprints. Stage validated dependency groups privately on target volumes with supported metadata, schema/domain/assets/credential policy and space checks, without writing live targets. Use existing archive/owner/SQLite/native types and helpers. TDD the original target-unverified regression and named focused filesystem scenarios; no publication, activation, UI, full suite or new feature scope. Record concrete limitations, review and verification before Done.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-17)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task17 implementation in progress in restore_plan.py/staging.py/test_restore_plan.py under ADR126. Original target_unverified regression recorded import and behavioral red then green. Latest focused cohort31passed2.45s, full new-scope Ruff clean, preliminary production Bandit0. Explicit local usernames are now required for selected configs, exact typed selector mappings verified against installed inert owner discovery. Source-derived synthetic roots, optional omissions, physical shared file candidates, SQLite WAL/SHM retirement scope, metadata disclosures, partial/dependency selection and target fingerprints are covered. Root-owned upstream producer metadata and restore-specific asset validation seams are coordinated separately; full asset-group integration and final report/checks pending. No publication/activation/UI or commits; do not mark Done yet.
Original Task17 asset dependency prerequisite: explicit restore-only logical/relative topology validation added to model recipes, persona/visual roots and recovered-media catalog/payload roles. Old capture validation remains unchanged. Model byte/dependency edges are checked, omitted bytes remain inert recipes; catalog raw payloads are not treated as SQLite. Agent realdata red3fail→green11, additional undeclaredmodel-edge red→fixed; final agent16newpass and26affectedold/newpass. Root independently reran16new:16passed18.95s (/private/tmp/chatbook-root-restore-assets-green.log), production fatalRuffclean/Bandit0. Added exact C82 read-only SQLite callsite inventory; literal-owner/policy checks2pass, roworder initiallyfailed then corrected and rerun. Report /private/tmp/chatbook-restore-asset-dependencies-report.md; ADR-126. Planner/stager integration and review corrections still in progress, no Task17 completion claim.
C82 inventory qualification finalized: documented newsite now follows C81 in stable order; exact expected connection-site range deliberately extended to82 with site rationale. Final stable-ID guard1passed1.81s (/private/tmp/chatbook-restore-assets-sqlite-ids-final.log); prior literal-owner and policy-link guards2passed. No guard bypass or broad connection allowance.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->