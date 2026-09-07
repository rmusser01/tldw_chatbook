---
id: TASK-399.5
title: A4 Complete read-only workflows and Database parity
status: To Do
assignee: []
created_date: '2026-07-23 14:23'
labels:
  - notes
  - library
  - integration
dependencies:
  - TASK-399.4
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
  - backlog/decisions/003-settings-library-rag-defaults.md
  - backlog/decisions/004-settings-storage-defaults-restart-boundary.md
  - backlog/decisions/008-sync-v2-client-m1-contract-alignment.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/015-shell-destination-ia.md
parent_task_id: TASK-399
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the useful read-only milestone with trustworthy export and Console handoff while proving existing Database Notes still work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can preview/copy the body and copy/export exact saved bytes only after a fresh raw-hash validation.
- [ ] #2 Console handoff stages the selection or an explicitly confirmed maximum of 80,000 characters, reports truncation, and sends no absolute path.
- [ ] #3 The A3 pageable Database adapter is verified with more than 100 notes and does not replace existing Database CRUD/filter services.
- [ ] #4 Database create, edit/autosave/conflict, search, keyword/link, template/import/export, RAG, MCP, remote Sync v2, and legacy sync retain parity with an active file root.
- [ ] #5 File projections never appear in Database-note counts, FTS, export, relations, RAG, MCP, or Sync payloads.
- [ ] #6 With zero linked roots, Notes retains prior Database behavior apart from `Link notes folder`; `Detached folders` appears only for retained projection evidence, while detected recovery evidence shows a source-scoped upgrade diagnostic rather than B1 list/verify/export controls.
- [ ] #7 Cross-source leave guards/Back restoration, File narrow Escape, partial errors, deterministic `PageDown`/load-more paging, focus, preview, export, and Console integration tests pass; delegated Database Escape retains existing behavior. A Library → Console → Library test proves the complete File workbench route, selection, expansion, focus target, and scroll anchor are restored.
- [ ] #8 A fixed Git smoke fixture confirms read-only Library activity produces no working-tree changes.
- [ ] #9 Read-only Unlink stops monitoring/election without touching files and discloses retained plaintext cache/FTS; Relink reuses identity only after root/inventory verification; Forget removes root, projections, and triggerless FTS rows with no source-file change. Forget explicitly states that removal from Chatbook queries is not secure erasure of database pages, storage media, or previously retained recovery bytes.
- [ ] #10 Existing Settings database backup/restore behavior remains unchanged and its UI explicitly excludes linked source files, file_notes.db, and notes_recovery.db.
- [ ] #11 With no active root, a `Detached folders` entry lists retained roots and sizes and offers Relink and Forget. An unavailable source produces an actionable Relink result without losing its detached record, while Forget remains usable and removes all queryable rows without requiring the original folder or volume to be present.
- [ ] #12 Public Link, Files-source, workbench, and detached-management controls remain default-off until the packaged A0-A4 integration, parity, performance, and Git read-only tests pass together; A4 owns the atomic A release-gate decision.
<!-- AC:END -->
