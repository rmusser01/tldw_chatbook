---
id: TASK-19006
title: Coordinate one active process per sync root
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:43'
updated_date: '2026-08-21 03:53'
labels:
  - notes
  - sync
  - lifecycle
dependencies:
  - TASK-19004
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give each lasting root one cross-process watcher and mutation owner, expose passive ownership honestly, and integrate fail-closed lease acquisition and release with application lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical root validation rejects overlap with lasting roots, Folder Files roots, application-private paths, symlink roots, and unsupported writable filesystems.
- [x] #2 An owner-private OS lock grants watcher and mutation authority to exactly one process per root; database lease rows are status only.
- [x] #3 Processes that cannot acquire authority expose `Passive in this process` and cannot reconcile, watch, or write.
- [x] #4 Closing admission blocks new work, lets admitted work reach a durable stage, and releases the lease only after settlement.
- [x] #5 Multi-process tests prove exactly one mutator and prove forced process death releases operating-system ownership.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write RED validation and real two-process lease tests covering overlaps, symlinks, passive ownership, forced death, and lifecycle ordering.
2. Implement the minimal owner-private OS-lock coordinator with canonical-root admission; keep database lease state diagnostic only.
3. Prove passive processes cannot watch, plan, or mutate and prove close-admission settles admitted work before release.
4. Run the focused process/coordinator gates, static checks, independent review, and task hygiene.

ADR required: no new ADR
ADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: ADR-059/073 already require one cross-process root owner, OS-lock authority, overlap refusal, passive state, and lifecycle fencing; this task implements that coordinator without activating runtime sync.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a minimal `portalocker` coordinator whose owner-private 0700 directory and 0600 inode-backed lock files grant the only process authority. Canonical root identity makes case aliases converge; symlink/reparse, overlap, private-root, offline, and unsupported-writer admission fail closed.
- Added path-free owner/passive/offline/rejected projections and authority gates. Root, lock-directory, lock-file, handle, mode, owner, and link identities are revalidated so replacement or root recreation revokes stale leases.
- Serialized settlement and release with explicit in-progress/committed states: admission closes first, settlement runs once, concurrent release waits, and close returns only after OS unlock/close success or a shared bounded failure. Forced process death releases the OS authority.
- Commits: `74a102196`, `c306aabc4`, `4bc78975b`. Final coordinator/process gate: 32 passed, 1 dependency warning; adjacent containment/filesystem gate: 59 passed. Ruff, formatting, and diff checks passed. Independent final review: Ready with no findings.
- ADR check: no new ADR; implementation follows ADR-059 and ADR-073.
<!-- SECTION:NOTES:END -->
