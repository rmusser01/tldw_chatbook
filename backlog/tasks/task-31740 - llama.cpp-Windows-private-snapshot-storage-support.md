---
id: TASK-31740
title: llama.cpp Windows private snapshot storage support
status: To Do
assignee: []
created_date: '2026-09-05 19:55'
labels:
  - llamacpp
  - snapshots
  - windows
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on TASK-31552 and PR #2419: provide a Windows private-storage implementation that can safely enable the manual snapshot manager. The current POSIX ownership checks intentionally fail closed on unsupported platforms. This work is deferred; simply bypassing those checks is not support. Before implementation, record the Windows privacy and filesystem contract in an ADR linked to ADR-029 and ADR-119, including any dependency choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Supported Windows configurations enforce and verify private snapshot catalog and per-launch working-directory access; unsafe or unverifiable permissions leave snapshots unavailable without preventing ordinary snapshot-disabled launches.
- [ ] #2 Path substitution, reparse-point, hard-link, and concurrent-process cases cannot publish, restore, prune, or delete files outside verified Chatbook-owned storage.
- [ ] #3 Atomic publication, integrity verification before restore, cross-process locking, retention, explicit deletion, interrupted-operation handling, and honest residual-file reporting preserve the existing manual manager contract on Windows.
- [ ] #4 Targeted tests and real Windows filesystem/server UAT verify the claimed access controls and save-restart-restore lifecycle; mocked POSIX checks alone do not count as Windows evidence.
- [ ] #5 Documentation states supported Windows environments, setup and recovery instructions, unsupported cases, and privacy limitations; existing POSIX behavior remains covered by targeted regressions.
<!-- AC:END -->

## References

- [Completed manual manager](task-31552%20-%20llama.cpp-manual-prompt-cache-snapshot-manager.md)
- [Merged PR #2419](https://github.com/rmusser01/tldw_chatbook/pull/2419)
- [ADR-119: snapshot ownership](../decisions/119-llamacpp-prompt-cache-snapshot-ownership.md)
- [Live UAT and qualification limits](../../Docs/superpowers/reviews/2026-09-05-llamacpp-slot-snapshots-uat.md)
