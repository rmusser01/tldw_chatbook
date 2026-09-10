---
id: TASK-31999
title: Write verify and publish recovery archives without overwrite
status: In Progress
assignee: []
created_date: 2026-09-07 23:57
labels:
- backup-recovery
dependencies:
- task-31978
- task-31985
- task-31995
- task-31998
updated_date: 2026-09-10 19:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Archives created by the writer pass the same bounded reader and carry truthful coverage, directory, and credential metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Archives created by the writer pass the same bounded reader and carry truthful coverage, directory, and credential metadata.
- [ ] #2 Existing backups/source/control files remain unchanged under races and aliases.
- [ ] #3 Failures and cancellation never expose incomplete output as a verified archive.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement existing ADR126/component03 Task16: existing-output preservation regression; bounded regular ZIP64 with strict manifest/topology and metadata; validate output/source/control alias boundaries, private destination-volume staging, optional/required encryption; verify with actual bounded reader before native publish_new; test race, aliases, corruption, cancellation, wrong password and ENOSPC; scoped static/review/commit only after capture integration. CaptureResult root/inventory/manifest_bytes contract is fixed by original Task15.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-16)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Agreed implementation contract: CaptureResult retains original source paths and actual used recovery control roots as intentionally excluded authority items; payloads live beneath capture.root under strict manifest payload names. Writer rejects source/control/capture aliases before reading and before native publication, including custom inventoried control roots. Published encrypted SealedArchive digest identifies ciphertext; actual reader.acquire(password) verifies a separate private decrypted candidate before publication, and verify_sealed applies to that acquired candidate. Capture orchestrator integration remains pending.
Final combined146-test cohort passes including plaintext/native encrypted publication through actual reader verification and existing age helper; no-overwrite/alias/cancel checks retained. Existing offline Go module/build caches used, no network or new credentials. New writer full Ruff/format clean and Bandit0. This verifies the writer slice, not whole backup or replacement readiness.
Original Task17 integration exposed missing manifest producer metadata: directory/exclusion ownership, dependencies/sharing and file mode/mtime were not serialized. Root will add strict inert ProducerItem records and optional file metadata to the existing archive schema, with current capture always populating them; legacy component archives remain readable but cannot claim verified replacement provenance. Reader checks exact coverage/status/owner/dependency/shared-byte consistency. No archive locator gains destination authority. This is necessary original restore-plan input, not a new recovery feature.
Added original restore-input metadata: strict inert producer inventory for files/directories/exclusions, dependencies and sharing, plus file mode/mtime. Capture emits synthetic parent/material records explicitly and retains excluded-owner identity without claiming an observed schema. Reader rejects mismatched coverage/status/owner, unknown dependencies, duplicate records, conflicting shared file bytes and coherent archives containing unsupported/unavailable/missing-required producers. Legacy archives remain readable with metadata absent; no local authority inferred. Red metadata4 failed7 passed; integrated reader/writer/capture114 passed47s; final contradiction red then metadata12 passed1.03s. New/changed modules Ruff clean except unchanged reader2→2 baseline; Bandit0. Restore planner consumes producer graph transitively; source/target mappings remain local.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->