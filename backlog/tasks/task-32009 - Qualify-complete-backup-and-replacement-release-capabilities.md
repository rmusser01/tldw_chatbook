---
id: TASK-32009
title: Qualify complete backup and replacement release capabilities
status: In Progress
assignee: []
created_date: 2026-09-08 00:03
labels:
- backup-recovery
dependencies:
- task-31978
- task-31985
- task-31993
- task-31994
- task-31995
- task-31996
- task-31997
- task-31998
- task-31999
- task-32000
- task-32001
- task-32002
- task-32003
- task-32004
- task-32005
- task-32006
- task-32007
- task-32008
updated_date: 2026-09-11 12:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Complete and replacement capability labels are backed by end-to-end owner, archive, native, and product evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Complete and replacement capability labels are backed by end-to-end owner, archive, native, and product evidence.
- [ ] #2 Both restore destinations and later rollback preserve expected data under ordinary and interrupted operations.
- [ ] #3 User/release documentation states qualified platforms, exclusions, credential limits, and recovery actions without overstating guarantees.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original Task26 only (Docs/superpowers/plans/2026-09-07-backup-recovery-06-release-evidence.md). Stages: (1) document implemented user flows and current evidence without release claims; (2) finish actual complete/isolated/replacement/later-rollback product qualification and synthetic owner fixture; (3) qualify native protocol/platform capability gates and named crash/adversarial cases; (4) update packaging, CI and user help, run named feature/inventory/lifecycle checks, review and record exact evidence. Dependencies remain unfinished; no acceptance criteria checked.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-06-release-evidence.md#task-26)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root begins bounded documentation slice: Docs/Backup-and-Recovery.md only, verified against actual F9 screen and startup-independent launcher. No capability flag changes or broad test runs. Current completed F9 replacement regression is commit3afae0913; later-rollback UI probe currently refuses before mutation and is under bounded diagnosis. User documentation must not imply whole Task26 or cross-platform qualification.
Documentation-only first slice drafted Docs/Backup-and-Recovery.md against actual F9/launcher/native_qualification.json. Independent source review /private/tmp/chatbook-user-docs-independent-review.md found missing recover --ask-password instruction; corrected with concrete rollback example and finish/abort distinction. Relative links and diff whitespace verified. Documents credential exclusions/password loss/private plaintext staging, explicit destinations/restart/safety copy, interrupted versus later rollback, retained copies, inert extraction, owner setup and currently unqualified platforms/product flow. No test/Bandit run for prose-only edit, no release flags changed; later rollback and Task26 acceptance remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->