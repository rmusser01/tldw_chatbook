---
id: TASK-399.7.1
title: B1a1 Pair recovery storage and acquire writable ownership
status: Won't Do
assignee: []
created_date: '2026-07-23 15:35'
updated_date: '2026-10-04'
labels:
  - notes
  - filesystem
  - recovery
dependencies:
  - TASK-399.5
  - TASK-399.6
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399.7
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the fail-closed recovery pairing, packaged capability admission, lease ownership, and capacity substrate required before any File Notes mutation can be admitted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Writable admission loads and validates the installed versioned APFS capability manifest, including schema, checksum, exact release/build result, native mutation-adapter ABI/artifact SHA-256, and probe artifact identity; missing, malformed, mismatched, or no-go evidence keeps File Notes read-only, while the recorded application commit remains non-gating provenance.
- [ ] #2 First pairing creates owner-only recovery storage only after proving complete prior absence, durably writes and verifies a versioned checksummed bootstrap marker through the B0-tested file and parent-directory flush sequence, commits recovery identity before projection identity, verifies the pair, and removes the marker durably.
- [ ] #3 Existing, orphaned, corrupt, incompatible, missing, or identity-mismatched projection, recovery, sidecar, or marker evidence is preserved and blocks mutation rather than being adopted, replaced, or silently rebuilt.
- [ ] #4 Read/write upgrade drains cooperative shared holders and acquires coordinator election plus exclusive mutation ownership; while held, every legacy filesystem pass is blocked in every process and passive processes start no watcher, reconciler, or file command.
- [ ] #5 Recovery admission enforces the fixed 1 GiB logical-retention cap, 64 KiB encoded-manifest ceiling, next-operation reservation, and at least 256 MiB remaining free space on the physical store volume.
- [ ] #6 Storage files, sidecars, markers, lease metadata, and temporary recovery artifacts use the specified owner-only permissions and fixed runtime namespace, with startup diagnostics that expose no note content or absolute main-database path.
- [ ] #7 This child exposes no writable control or read/write mode transition.
<!-- AC:END -->

## Closeout 2026-10-04

Closed superseded, not implemented as written. Every substrate this child was
to pair — installed APFS capability-manifest admission, coordinator election,
exclusive mutation leases, and reserved recovery capacity — is ADR-021
machinery that ADR-029 rejected; no manifest, lease, or fixed recovery store
was ever built. Recovery under the shipped design is the root-namespaced
SQLite replica's current bytes plus protected-path checkpoints
(`tldw_chatbook/Notes/file_notes_replica.py`; TASK-969, PR #992).

Superseded by: ADR-029 (backlog/decisions/029-file-notes-disk-authority.md) via TASK-969 (PR #992).
