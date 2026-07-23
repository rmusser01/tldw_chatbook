---
id: TASK-399
title: >-
  File-backed Notes disk-authoritative Library management and local recovery
  replica
status: To Do
assignee: []
created_date: '2026-07-23 04:06'
updated_date: '2026-07-23 05:07'
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
Roll-up tracker for phased first-class management of one existing Git-managed Markdown/text root in Library Notes. Disk and filename/path remain authoritative, Git stays external, isolated main-database tables provide the derived file projection/search, and an independent same-device recovery database protects mutations and selected files/folders. This tracker is never implemented as one PR or moved into implementation as a single unit: after the revised design is approved, planning must create atomic child tasks in A, B1, B2, then B3 dependency order. Linux/Windows writes, additional active roots, folder mutation, file templates/keywords/links/MCP/RAG, mixed bulk export, configurable recovery quotas, general purge, recovery-store relocation/clone, database-pair backup/restore, recovery-only in-place restore, and Git controls are separately approved follow-ups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Milestone A lets users preview and link one ordinary local root read-only on packaged macOS, Linux, and Windows builds; browse/search it through isolated file projection/FTS tables; monitor external changes; preview, exact-copy/export, and hand off to Console; and use deep/5,000-sibling fixtures without altering source bytes or requiring notes_recovery.db.
- [ ] #2 Filename and relative path remain canonical; opaque frontmatter, BOM, uniform LF/CRLF style, and final-newline facts round-trip exactly while the body is edited separately; mixed/lone-CR normalization requires hash-bound acknowledgment and a verified recovery copy; and existing Database-note schema, triggers, CRUD, search, export, keyword/link, RAG, MCP, and Sync behavior remain unchanged. Zero-root UI/runtime differs only by the specified Link/Recovery actions, and active-root Database selections use the existing Database-note implementation.
- [ ] #3 Milestone B1 enables debounced hash-checked create, body save, rename, and move only in existing directories on a verified local macOS/APFS root; every mutation is journaled, round-trip verifies required safety bytes before disk mutation, preserves an atomically displaced target, reconciles externally changed bytes near real time, and records only successful Chatbook working-tree changes in the in-memory current-session view.
- [ ] #4 Milestone B2 confirms against the current exact file, records and verifies a self-contained deletion snapshot/revision and tombstone, deletes the actual file through quarantine, retains its exact bytes for at least 30 days, and provides a working minimal restore before Delete is exposed.
- [ ] #5 Milestone B3 lets users protect selected files or folder prefixes in independent notes_recovery.db current replicas, coalesce editing checkpoints, use minimal per-note history/restore, and enumerate/verify/exact-export retained content without the main database; main/recovery storage-instance IDs and a shared random recovery-instance UUID must match, and fixed capacity/free-space admission fails closed without silently replacing the store or evicting guaranteed/unresolved content.
- [ ] #6 FileNotesRepository and the combined read seam keep file projections out of Database-note write paths; remote Sync v2, generic Database-note CRUD/export, MCP, RAG, and keyword/relation triggers cannot mutate them or disk, while pre-activation cooperative legacy passes hold a cross-process shared root-mutation lease and activation holds it exclusively after draining them; passive processes then run no legacy filesystem sync, and every legacy engine entry point also uses the in-process canonical-root gate.
- [ ] #7 A separate FileNotesWorkbench and file-only FileNotesSessionController deliver the approved collapsible tree/search-replacement editor UX, path breadcrumb, two-line prioritized save/capability status, file actions, wide split layout, narrow Navigator/Editor switching without state loss, explicit source labels, unchanged external Git workflow, and delegation of Database-source selections to the existing LibraryNotesCanvas/handlers.
- [ ] #8 Capability, mandatory pre-constructor online-backup/additive-migration ordering and Database-only previous-schema fallback, recovery pairing/relocation failure, one-root shared/exclusive OS lease, pinned-root/nested-mount containment, APFS durability, recovery corruption/capacity, conflict/crash, watcher-storm/polling, performance, accessibility/focus, active-root Database parity, and zero-configuration non-degradation tests pass; packaged Linux/Windows tests prove writable actions remain explicitly unavailable until native adapters are separately approved.
<!-- AC:END -->
