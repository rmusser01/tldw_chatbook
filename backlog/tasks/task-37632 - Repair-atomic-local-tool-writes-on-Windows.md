---
id: TASK-37632
title: Repair atomic local tool writes on Windows
status: Done
created_date: 2026-10-29 22:34
assignee:
- '@codex'
updated_date: 2026-10-29 23:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow valid Windows file-tool writes to complete with the existing pinned-directory and stale-write guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Windows fs_write creates and replaces files through native retained directory handles.
- [x] #2 Stale CAS, root replacement, reparse and sensitive-path controls remain enforced.
- [x] #3 POSIX and remote stdlib worker behavior retain their existing contracts.
- [x] #4 Real worker tests, focused atomic-write tests, lint and documentation are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md. Reason: refine search outcomes and optional listing worker arguments while retaining ADR-078/094/101 authority. Follow Docs/superpowers/plans/2026-10-29-cli-tool-recovery.md with focused red/green tests, implementation, documentation, targeted checks and fresh review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Windows write/read/CAS helpers reuse native retained directory operations. CAS reads through the locked target handle; explicit FileRenameInformationEx/POSIX replacement keeps the lock through the atomic swap. Ordinary native rename behavior and POSIX paths stay intact; Windows creation/replacement retains the facade private-DACL policy. Added real worker create/read/replace, stale/absent, cleanup, inode substitution, lock-retention and junction-swap controls. Exact-byte and POSIX-mode fixtures were corrected for Windows; the cross-process lock probe uses the worker admitted-root body. Evidence: windows-green-scoped 36 passed/1 POSIX-mode skip; native facade controls passed in the root selection; new native tests and lint pass. Six existing file-symlink fixtures fail at setup WinError1314, with real native junction refusal covered. Remote artifact regenerated and checked; actual POSIX execution unavailable on this host. ADR-254 and architecture/lessons docs record the native contract.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
