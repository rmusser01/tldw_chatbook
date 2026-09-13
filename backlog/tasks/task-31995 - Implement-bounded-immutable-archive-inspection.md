---
id: TASK-31995
title: Implement bounded immutable archive inspection
status: Done
assignee: []
created_date: 2026-09-07 23:54
labels:
- backup-recovery
dependencies:
- task-31978
- task-31984
- task-31985
- task-31986
- task-31987
updated_date: 2026-09-10 14:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Inspection uses completed immutable input bound to a digest and validates manifest/payload integrity under bounded resource use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inspection uses completed immutable input bound to a digest and validates manifest/payload integrity under bounded resource use.
- [x] #2 Malformed, ambiguous, hostile, or unsupported archives cannot select destination paths or execute content.
- [x] #3 Limits apply during source copy and decryption before untrusted manifest information is available.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement the existing task-12 steps in Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-12. Reuse ADR-126 and existing crypto/private path primitives. Add regression tests first, implement bounded immutable source acquisition and strict ZIP64/manifest inspection, verify adversarial and encrypted cases with focused tests, run scoped lint and Bandit, then record evidence. Scope is the original plan, with no added recovery features.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-12)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented original Task 12 using ADR-126: strict frozen manifest records, privately copied and digest-bound archive acquisition, authenticated decryption before ZIP inspection, independent streaming container quotas, ZIP64 structural/payload verification, explicit compression review binding, and sanitized inert metadata. Added the writer/reader contract in backlog/docs/backup-recovery-archives.md. Corrected both scoped review findings with behavioral regressions: validate full bounded deflate streams and track successful staging ownership locally. Verification: 85 passed with `GOMODCACHE=/private/tmp/task31985-go/mod GOCACHE=/private/tmp/chatbook-task12-gocache GOPROXY=off python -m pytest Tests/Backup_Recovery/test_archive_reader.py Tests/Backup_Recovery/test_crypto.py -q --tb=short --basetemp=/private/tmp/chatbook-task12-all-affected` after activating the project venv. Scoped Ruff/format and git diff --check passed. Bandit has only the same two LOW crypto subprocess findings reproduced at clean HEAD; no new findings or suppressions. Independent spec and quality re-review approved both fixes. Reports: /private/tmp/chatbook-backup-task12-report.md and /private/tmp/chatbook-backup-task12-review.md. Qualification remains this original slice only: release-helper shipping and downstream destination/owner validation belong to later tasks; no Complete backup is exposed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed bounded archive inspection and its strict manifest/ZIP contract, with 85 passing affected tests and approved scoped review. Original task-12 scope only; remaining backup/restore tasks are still pending.
<!-- SECTION:FINAL_SUMMARY:END -->
