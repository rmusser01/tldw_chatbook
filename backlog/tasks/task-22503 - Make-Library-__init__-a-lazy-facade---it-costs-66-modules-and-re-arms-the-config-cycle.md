---
id: TASK-22503
title: >-
  Make Library/__init__ a lazy facade - it costs 66 modules and re-arms the config cycle
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - architecture
  - startup
priority: medium
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Found while fixing TASK-22223: the entire cost of config importing a Library leaf was
`Library/__init__` executing its service stack — `library_collections_service` →
`library_collections_state` → `Sync_Interop/__init__` → `Chat/__init__` →
`server_chat_conversation_service` → `runtime_policy.bootstrap`. 22223 moved the one
normalizer to a stdlib-only Utils leaf (config closure 106 → 40 modules) and killed the live
circular import, but ANY `tldw_chatbook.Library.*` import still pays ~66 modules and re-arms
the same cycle class for the next importer.

The `TTS/__init__` PEP 562 lazy facade from the TASK-21108 era is the in-repo precedent.

## Acceptance Criteria

- [ ] Importing a single `tldw_chatbook.Library.*` leaf does not execute the collections/sync/chat service stack (subprocess census)
- [ ] The public names `Library/__init__` exports still resolve (PEP 562 `__getattr__`), with the eager-import consumers unchanged
- [ ] A guard pins the leaf-import closure so the stack cannot creep back
