---
id: TASK-31987
title: Implement stable maintenance admission and native storage qualification
status: Done
assignee:
  - codex
created_date: '2026-09-07 23:49'
updated_date: '2026-09-08 04:30'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31986
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Participating processes cannot mutate admitted namespaces during maintenance, including aliases and inode replacement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Participating processes cannot mutate admitted namespaces during maintenance, including aliases and inode replacement.
- [x] #2 Deadlock, timeout, cancellation, and crashed-holder cases preserve data and durable recovery evidence.
- [x] #3 Native publication cannot overwrite an existing artifact, and unsupported storage returns an explicit unavailable capability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved native storage and admission contract; reuse ADR-126.

1. Establish importable behavioral RED publication regression.
2. Implement stable namespace registration, verified aliases, reserved remapping, and native cross-process admission outside managed roots.
3. Close mutation admission before draining established participants; order acquisition and preserve safe retirement on timeout/cancellation.
4. Implement private creation and qualified durable native no-replace publication with pinned parents; refuse unsupported operations/filesystems.
5. Exercise independent child processes for aliases, inode replacement, remapping, retirement, death, stale evidence, contention and native publication failures.
6. Run named focused/native tests, exact persistence census, affected legacy advisory-lock guard, scoped lint/format and diff checks.
7. Self-review, record exact evidence and interfaces, and commit scoped changes. Leave In Progress and AC unchecked for controller review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-126 stable admission and native publication primitives. Admission registers verified roots and aliases, uses stable lock inodes and ordered gates/leases, drains without holding registry authority, and preserves remap reservations across timeout, cancellation and process death. Known incompatible-client evidence has an OS lifetime; the legacy instance lock remains advisory. Admission coordinates one verified control root; startup must establish shared authority for aliases.

Native publication uses pinned-parent Darwin renameatx_np(RENAME_EXCL), validates private regular-file/directory payloads and never falls back to overwrite. Files and directory metadata receive fsync plus F_FULLFSYNC barriers. Registry replacement first publishes a bounded versioned registry.pending.json with before/after states and a write identity, retains it until durable completion, and refuses admission under uncertain intent. Initial creation cannot synthesize a disappeared existing registry. Explicit recovery reconciliation remains a later journal responsibility.

Installed strict native qualification protocol 2 rejects old fsync-only protocol 1, missing/unsupported protocols, malformed/unknown fields and invalid operation lists. Actual qualification is Darwin 25.5.0 / arm64 / Python 3.12.11 / APFS; only tested publication/admission primitives are enabled. Other identities and overall isolated/replacement capabilities remain unavailable. Source/wheel inventories require native_qualification.json. The producer census now includes actual native rename, OS replace and intent-removal seams.

Independent review identified missing post-publication full flush and permissive qualification metadata. Fix commit 276931e77 addresses both; scoped re-review reports all findings addressed and no new breakage. The fix also demonstrates independent-process refusal after final-remap barrier failure and safe intent-cleanup behavior.

Final covering command: python -m pytest Tests/Backup_Recovery/test_native_files.py Tests/Backup_Recovery/test_admission.py -q --tb=short (65 passed, 12.46s). Final python -m pytest Tests/Architecture/test_backup_owner_inventory.py -q (11 passed, 11.60s). Before the fix, directly affected legacy lock tests passed 10 and the distribution-contract guard passed 1; their behavior/resource inclusion did not change in the fix. Scoped Ruff fatal checks, six-file formatting and git diff --check passed. Behavioral RED/GREEN reproduced existing-target overwrite failure, nested alias admission, linked payload, post-publication flush omission, malformed qualification, final-remap admission reopening and disappeared-registry synthesis.

All evidence used the shared Python 3.12.11 interpreter read-only and private fixtures. Real processes tested aliases, inode replacement, SQLite retirement, contention, timeout/cancel, death and crash reservations; native tests covered file/empty/populated-directory publication, races and barriers. Process-exit/flush evidence does not claim physical power-cut testing. Existing dependency/AST warnings remain recorded for final review; no full suite or remote workflow ran.

Files: Backup_Recovery/{admission,native_files,qualification}.py and native_qualification.json; native/admission/architecture tests; advisory-lock documentation; packaging inventories/checker/assertion; owner census; backlog/docs/backup-recovery-native-admission.md. Commits: bfe8b6b18 and 276931e77. ADR: backlog/decisions/126-complete-local-backup-and-recovery.md. No new ADR, unrelated changes, push or merge.
<!-- SECTION:NOTES:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-4)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.
