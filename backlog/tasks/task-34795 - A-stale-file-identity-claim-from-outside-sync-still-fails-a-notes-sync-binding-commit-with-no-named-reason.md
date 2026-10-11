---
id: TASK-34795
title: A stale file-identity claim from outside sync still fails a notes sync binding
  commit with no named reason
status: To Do
created_date: 2026-10-11 17:52
labels:
- notes-sync
priority: high
updated_date: 2026-10-11 21:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The notes sync store allows one active binding per file identity, which is the device and inode number. A binding keeps the identity of a file that was replaced by rename or deleted outside the app until its folder's next pass. On a filesystem that hands a freed inode number to the next new file, such as ext4, a file created outside the app can take a number a stale binding still records. Binding it then breaks the unique index and the operation ends as an executor failure left for Recovery. TASK-34789 fixed this for files that sync itself writes, by never taking a claimed number; it cannot cover a file sync did not create. Recovery cannot clear it either: resolving the cleanup raises the same integrity error and the folder stays held. Any tool that rewrites several bound files by rename (an editor's save, a version-control checkout, a file copier) can reach this on such a filesystem. The failure also carries no named reason: every untyped exception is recorded as an executor failure with no trace, which is why the CI flake behind TASK-34789 could not be read from any log.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A file created outside the app on an inode number that a stale binding still records can be bound without its operation failing
- [ ] #2 Two bindings are never active for the same live file
- [ ] #3 When a binding commit is refused for an identity collision, the recorded reason names it
- [ ] #4 Resolving the cleanup for an operation that failed on an identity collision does not raise, and the folder is released once the stale binding is gone
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
