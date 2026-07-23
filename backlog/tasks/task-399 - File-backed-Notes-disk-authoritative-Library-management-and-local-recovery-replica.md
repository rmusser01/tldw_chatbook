---
id: TASK-399
title: >-
  File-backed Notes disk-authoritative Library management and local recovery
  replica
status: To Do
assignee: []
created_date: '2026-07-23 04:06'
updated_date: '2026-07-23 14:24'
labels:
  - notes
  - library
  - storage
  - recovery
  - epic
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roll-up tracker for phased first-class management of one existing Git-managed Markdown/text root in Library Notes. Disk and filename/path remain authoritative and Git stays external. A dedicated `file_notes.db` provides the derived plaintext projection/search without changing ChaChaNotes; independent same-device `notes_recovery.db` protects mutations and selected files/folders. This epic is delivered only through its A0-A4, B0, B1a-B1b, B2, and B3a-B3b children. Linux/Windows writes, additional active roots, folder mutation, file templates/keywords/links/MCP/RAG, mixed bulk export, configurable recovery quotas, general purge, recovery-store relocation/clone, paired-store backup/restore, recovery-only in-place restore, and Git controls are separately approved follow-ups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Milestone A ships through A0-A4: one ordinary local root can be previewed/linked read-only on packaged macOS, Linux, and Windows; dedicated `file_notes.db` provides scalable path/body projection, triggerless retryable FTS, near-real-time reconciliation, selectable body reading, exact copy/export, Console handoff, and explicit plaintext-cache disclosure without changing source bytes or requiring recovery storage.
- [ ] #2 Filename/path remain canonical; opaque frontmatter, BOM, newline/final-newline state, and supported security facts round-trip exactly once writing ships. Mixed/lone-CR normalization requires hash-bound acknowledgment and verified prior bytes; files with unsupported ACL/xattr/ownership/flag metadata remain read-only rather than losing it.
- [ ] #3 B0 checks in a packaged-app APFS capability/version matrix with explicit go/no-go evidence for pinned traversal, exchange/no-replace, metadata handling, file/directory durability, full-fsync, and a named power-cut/reboot result on every supported macOS release before B1 begins.
- [ ] #4 B1a-B1b enable debounced hash-checked create/save/rename/move only in existing directories on a verified local APFS root. Disk/recovery/projection complete before the journal closes; FTS remains non-blocking; conflicts/startup are durable; and recovery-only enumerate/verify/exact-export works without opening ChaChaNotes or `file_notes.db`.
- [ ] #5 B2 confirms the exact file, verifies a self-contained deletion snapshot/tombstone, deletes through quarantine, retains exact bytes for at least 30 days, and restores only to the absent original path (or exact-exports when occupied/missing-parent) before Delete is exposed.
- [ ] #6 B3a-B3b protect selected files/folder prefixes with verified current replicas, fixed capacity/free-space admission, coalesced checkpoints, bounded history, and separately confirmed alternate-path/copy/overwrite restore without silently replacing stores or evicting guaranteed/unresolved content.
- [ ] #7 ChaChaNotes schema/version/constructors and Database-note backup/restore behavior plus CRUD/search/export/relations/RAG/MCP/Sync remain unchanged; backup labels explicitly exclude linked files and both File Notes databases. A uses only coordinator election; B1 mutation upgrade drains shared legacy holders and acquires exclusive ownership. Repository/source errors remain isolated and no File path can mutate through a Database-note service.
- [ ] #8 The workbench provides exact source authority labels, full-surface delegation to existing `LibraryNotesCanvas`, leave guards and return targets, source-grouped independent paging, unsafe-draft-first status, deterministic focus transitions, wide/narrow state preservation, zero-root parity, and the fixed scale/performance/crash/security test gates.
<!-- AC:END -->

## Child Tasks

- [TASK-399.1](task-399.1%20-%20A0-Isolate-file-note-projection-storage.md) — A0 storage/isolation
- [TASK-399.2](task-399.2%20-%20A1-Preview-and-link-one-read-only-notes-root.md) — A1 discovery/preview
- [TASK-399.3](task-399.3%20-%20A2-Project-search-and-reconcile-file-notes.md) — A2 projection/search/reconciliation
- [TASK-399.4](task-399.4%20-%20A3-Add-the-File-Notes-Library-workbench.md) — A3 workbench
- [TASK-399.5](task-399.5%20-%20A4-Complete-read-only-workflows-and-Database-parity.md) — A4 read-only workflows/parity
- [TASK-399.6](task-399.6%20-%20B0-Prove-the-macOS-APFS-writable-substrate.md) — B0 APFS proof
- [TASK-399.7](task-399.7%20-%20B1a-Build-journaled-create-save-and-autosave-foundation.md) — B1a gated write foundation
- [TASK-399.8](task-399.8%20-%20B1b-Add-move-conflict-resolution-and-startup-recovery.md) — B1b writable completion
- [TASK-399.9](task-399.9%20-%20B2-Delete-files-with-verified-minimal-restore.md) — B2 delete/minimal restore
- [TASK-399.10](task-399.10%20-%20B3a-Protect-selected-files-and-folders.md) — B3a protection
- [TASK-399.11](task-399.11%20-%20B3b-Add-coalesced-history-and-safe-restore-choices.md) — B3b history/expanded restore
