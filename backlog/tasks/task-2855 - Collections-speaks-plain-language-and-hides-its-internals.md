---
id: TASK-2855
title: Collections speaks plain language and hides its internals
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - collections
  - ux-copy
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-07, observed at dev `6ffa56516`). Owner ruling 2026-08-07: keep the
row, rewrite to plain language (not hide until adapters exist).

The Collections canvas shows internal spec/roadmap language to end users: "Item reader
readiness", "Authority: local", "Content use boundary", "Blocked later: item reader, Search/RAG,
Study, Console handoff, server sync", "Next: collection item adapters are required before
item-level actions unlock", "Write Sync Safety … Sync: dry-run only". The empty state renders
"No stored collection items are available locally yet." twice on one screen, and three helper
sentences repeat the same enable-Create rule (all three persist unchanged after a valid name is
typed). No surface anywhere offers "Add to collection", so the canvas can only name empty sets.

Related P3s folded in: the triple-redundant helper text and the duplicated empty-state sentence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The canvas's status copy is one plain-language line (e.g. "Collections hold saved items for review — adding items is coming; you can create and name collections now."); spec/architecture vocabulary (adapters, authority, content use boundary, blocked-later lists) no longer appears on the canvas
- [ ] #2 Sync-safety/internal detail moves behind the Details disclosure or is removed
- [ ] #3 The empty state renders its message once, and the enable-Create guidance is a single sentence that disappears (or updates) once a valid name is typed
- [ ] #4 Live TUI verification of empty and one-collection states
<!-- AC:END -->
