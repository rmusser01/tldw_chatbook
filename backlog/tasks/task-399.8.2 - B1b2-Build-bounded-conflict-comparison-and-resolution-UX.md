---
id: TASK-399.8.2
title: B1b2 Build bounded conflict comparison and resolution UX
status: To Do
assignee: []
created_date: '2026-07-23 15:36'
labels:
  - notes
  - filesystem
  - recovery
  - ui
dependencies:
  - TASK-399.8.1
documentation:
  - >-
    Docs/superpowers/specs/2026-07-22-file-backed-notes-authority-recovery-design.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
parent_task_id: TASK-399.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide a complete three-sided conflict experience that keeps base, draft, and disk identities durable and usable without freezing or discarding the active editor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focused clean editor whose file changes on disk keeps visible bytes stable and offers Reload from disk and Compare; typing first durably pins base and latest disk state before accepting input and creates actionable Attention.
- [ ] #2 Dirty-buffer external edits, moves, deletion, and late publication races durably retain base, draft, and latest disk sides before navigation can release the editor; absent disk is represented explicitly.
- [ ] #3 Compare names Base, Draft, and Disk, computes off the UI thread with cancellation and bounded input and output, shows paged unified diff on narrow terminals and optional labeled sides on wide terminals, reports elision, and falls back to hashes, sizes, exact copy, and exact export for oversized content.
- [ ] #4 Escape closes Compare and returns focus to its opener without resolving Attention; keyboard and screen-reader labels expose side identity and active resolution state.
- [ ] #5 Save draft as new note, Keep editing, Overwrite disk with draft, and Discard draft and load disk each re-hash current disk, preserve the only copy of every side, enforce their confirmations and journal protocols, and fail closed when prefix, style, metadata, or destination state is ambiguous.
- [ ] #6 Navigation, resize, Library to Console to Library reconstruction, source switch, root-offline transition, and app lifecycle preserve conflict visibility and never bypass the draft durability guard.
- [ ] #7 There is no automatic merge, timestamp winner, or silent reload of a dirty buffer; unresolved items remain in Needs attention.
- [ ] #8 This child exposes no public writable mode transition or final B1 controls.
<!-- AC:END -->
