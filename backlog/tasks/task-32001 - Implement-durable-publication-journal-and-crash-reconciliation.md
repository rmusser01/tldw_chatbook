---
id: TASK-32001
title: Implement durable publication journal and crash reconciliation
status: In Progress
assignee: []
created_date: 2026-09-07 23:58
labels:
- backup-recovery
dependencies:
- task-31978
- task-31987
- task-31988
- task-32000
updated_date: 2026-09-10 20:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Interrupted publication is classified using durable journal and actual filesystem evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Interrupted publication is classified using durable journal and actual filesystem evidence.
- [ ] #2 No supported startup opens an ambiguous mixed generation, and original/candidate/rollback evidence is retained.
- [ ] #3 Native crash tests cover every durable transition and refuse unqualified filesystem semantics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Original Task18, ADR-126 and Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-18.
1. Write focused behavioral-red journal tests, including incomplete publication remaining recovery-required.
2. Implement versioned private durable event records and validated evidence/state transitions for prepared, rollback_verified, publication_started, artifact_published, installed_validated, activation_recorded, committed, rollback_started and recovery_required.
3. Register fixed bootstrap pending pointer and affected namespaces before publication; preserve previous/candidate identities, target-volume paths, generation, rollback reference, retirement intent and progress without secrets.
4. Publish through qualified native no-replace and checked retirement/replacement primitives per target volume.
5. Reconcile actual filesystem identity/digests against intent after interruption; preserve uncertain or corrupt evidence and startup fencing.
6. Test real subprocess interruption at durable boundaries, rename-before-record, sidecars, failure and unavailable volumes; keep admission fenced until installed validation and activation are durable.
7. Run focused checks, Ruff/format, Bandit and review; update inventory/doc evidence and only mark Done when all original criteria pass.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-18)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
First original Task18 journal/evidence slice implemented (not complete publication): private bounded exclusive sequence records with stable native lock, strict prepared/publication intent schemas, chained record digests, retained corrupt/partial evidence, bounded nofollow file/tree identity+bytes observations and staged/published/uncertain classification. No admission clearing or successful commit inferred from labels; replacement start refuses absent verified rollback. Behavioral red6fail→basicgreen6; next evidence red3fail13pass→16pass; review caught missing retained/previous consistency and earlier-child mutation duringlater read, both reproduced red then fixed. Root latest18passed2.22s /private/tmp/chatbook-journal-review-green.log; Ruffclean/Bandit0 (/private/tmp/chatbook-journal-final-bandit.json). Two real child exits cover current durable preparation/intent boundaries and retain actual startup fence. Full Task18 state transitions, qualified publication/retirement and activation-proof integration remain unfinished; not marked Done. Review /private/tmp/chatbook-journal-records-review.md; ADR-126.
Future isolated selector registration verified: _verify_pending_selector preserves original existing-file fingerprint checks and accepts only genuinely absent leaf/descendants beneath a pinned nofollow ancestor with descriptor-relative absence recheck. No destination directories are created. Pending selector blocks startup both before and after later file creation. Agent new+bootstrap43passed8.72s; root independently reran final13new cases:13passed1.12s (/private/tmp/chatbook-root-future-selector-green.log). Newlinks/non-directory/unsafeancestor/overlap/race regressions included. Production Ruff3→3/noaddition, Bandit0→0. Source/report /private/tmp/chatbook-future-selector-report.md. Task18 publication integration remains in progress; no fence-clear authority added.
Original Task18 bounded publication slice now implemented and reviewed: durable successful-stage receipt binds verified sealed archive, exact plan and private candidate descriptor; locally supplied preparation checks fixed pending association, independent target fingerprint and actual affected namespace coverage. Raw file/tree replacement requires actual authenticated encrypted rollback with exact byte/name/metadata coverage; SQLite remains explicit owner-receipt-required for Task22. Native no-replace publication and original retirement retain crash evidence. Review found and fixed three concrete gaps: complete-but-unflushed journal/fence records on retry (native file+dir barriers reestablished), replaced parent directory after publication_started (actual existing/future-container identities bound and checked again inside pinned native handles), and unrelated enrolled-profile fence (actual registry scope coverage and affected namespaces checked). Root behavioral regressions2failed before scope/parentfix, additional native-parent gap1failed thenfixed. Final publication+native focusedcohort75passed17.25s (/private/tmp/chatbook-publication-review-final.log); follow-up7focused cases also pass including correctly enrolled positive publication. FullRuff/formatclean, productionBandit0; finalreview approved /private/tmp/chatbook-publication-review.md. Reports /private/tmp/chatbook-publication-report.md and /private/tmp/chatbook-publication-reviewed-bandit.json. Exact producer census updated. RemainsInProgress: 1MiB prepared/descriptor cap, SQLite held-owner rollbackreceipt, installedmetadata/validation, credentialapplication, activation/fenceclear and rollbackexecutor still outstanding. No Task18Done or end-to-end claim; ADR126.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->