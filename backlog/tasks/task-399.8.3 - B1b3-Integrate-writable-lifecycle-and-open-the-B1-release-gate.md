---
id: TASK-399.8.3
title: B1b3 Integrate writable lifecycle and open the B1 release gate
status: To Do
assignee: []
created_date: '2026-07-23 15:37'
labels:
  - notes
  - filesystem
  - recovery
  - ui
dependencies:
  - TASK-399.8.2
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate read/write mode transitions, safe Unlink and Forget lifecycle barriers, and the all-or-nothing release gate that finally exposes the completed B1 writable workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Read/write upgrade is offered only for a healthy paired root on an exact go entry in the installed B0 manifest and succeeds only after command admission, coordinator election, exclusive mutation ownership, complete legacy-pass drainage, recovery capacity, and metadata support are revalidated.
- [ ] #2 Read/write downgrade and Unlink close new operations, cross the mutation and editor barrier, durably retain or explicitly resolve every draft and Attention item, stop reconciliation, and release both lease layers before publishing detached state.
- [ ] #3 Detached folders remains reachable with no active root and offers Relink and Forget using retained root identity and projection evidence; source absence does not block Forget.
- [ ] #4 Forget is blocked by pending operations, Attention, unresolved drafts, or unexported sole copies, requires explicit confirmation, purges only the selected logical projection, FTS, and eligible recovery records, and states that SQLite deletion is not secure erasure.
- [ ] #5 Controlled app shutdown, Library reconstruction, root offline or identity change, recovery corruption, capacity loss, and passive-process transitions preserve the same barriers and keep Database Notes usable.
- [ ] #6 Create, save, autosave, rename, move, read/write mode, Unlink, and Forget controls remain default-off until TASK-399.7.1 through TASK-399.8.3 pass together and the final B1 release candidate reruns B0 against its exact packaged native adapter/probe on the two-release matrix; unsupported or unmatched systems remain read-only.
- [ ] #7 Large files above the interactive ceiling, unsupported manifests or security metadata, network or non-APFS roots, and ambiguous path identity remain read-only with exact copy or export rather than exposing a partial writable path.
- [ ] #8 Release tests prove no File path mutates through Database-note services, no legacy filesystem pass runs while File Notes owns exclusive mutation, and Git staging, commit, and push controls remain absent.
<!-- AC:END -->
