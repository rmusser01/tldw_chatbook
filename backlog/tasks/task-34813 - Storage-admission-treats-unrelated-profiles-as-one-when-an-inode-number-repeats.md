---
id: TASK-34813
title: Storage admission treats unrelated profiles as one when an inode number repeats
status: To Do
created_date: 2026-10-12 05:26
labels:
- backup-recovery
- correctness
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Admission decides which storage namespaces must not run at the same time by comparing identity tokens, and each root's token is its inode number alone. The device was removed from the token on purpose (TASK-34200), because macOS can renumber a volume across a reboot and the app then refused every start. The cost is that two unrelated roots are treated as one group whenever one of them holds an inode number the other ever had: on a filesystem that reuses freed inode numbers, or on two filesystems that number alike. The history a namespace keeps only grows, so one coincidence lasts. Verified on a scratch control root while investigating flaky tests for TASK-34789: after one namespace's history contained the other's inode number, ordinary use of the first waited for the whole time the second was under maintenance, and maintenance of the second timed out while the first was in use, with no storage shared. Whether a Linux filesystem produces the collision in practice was inferred, not run. Several recovery checks intersect a config file's inode number with other namespaces' history in the same way, and the config file gets a new inode on every save.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two namespaces whose roots share no storage are not grouped because one holds an inode number the other once had
- [ ] #2 A profile whose volume was renumbered across a reboot still starts (the TASK-34200 behaviour is kept)
- [ ] #3 A test builds the collision without relying on a particular filesystem and fails before the fix
- [ ] #4 The recovery checks that compare a config file's identity with other namespaces' history are covered by the same rule or listed with the reason they are not
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
