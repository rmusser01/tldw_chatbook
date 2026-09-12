---
id: TASK-32496
title: Make complete backup and recovery work on macOS Linux and Windows
status: In Progress
created_date: 2026-09-12 16:36
labels:
- backup-recovery
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the backup implementation's platform-specific filesystem and release assumptions so the existing complete backup/restore/replacement/rollback behavior works on all three application platforms. User explicitly requires actual Linux testing and authorizes GitHub Actions Windows testing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing complete backup, isolated restore, replacement and retained-copy rollback have working platform operations on macOS, Linux and Windows without Go.
- [ ] #2 Actual installed application workflows create archives and restore/verify synthetic data on the supplied Linux host and native Windows GitHub runner; macOS covering regression tests pass. No component-only result is represented as product success.
- [ ] #3 Preserve private storage, no-overwrite publication, cooperative locking, authentication, cancellation and interruption recovery; update PR2642 with exact tests, limitations and source identities.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce actual Linux/Windows product failures and map platform dependencies.
2. Implement platform-native filesystem/locking/private-path support with targeted regressions, preserving existing recovery contracts.
3. Run complete installed backup/restore/replacement/rollback workflows on Linux over SSH, Windows GitHub Actions and macOS; fix observed failures.
4. Independent review, scoped security/static checks, update ADR126 correction and PR evidence. No completion claim until real product verification.
ADR: update existing backlog/decisions/126-complete-local-backup-and-recovery.md for cross-platform platform contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
