---
id: TASK-22503
title: >-
  Make Library/__init__ a lazy facade - it costs 66 modules and re-arms the config cycle
status: Done
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

- [x] Importing a single `tldw_chatbook.Library.*` leaf does not execute the collections/sync/chat service stack (subprocess census)
- [x] The public names `Library/__init__` exports still resolve (PEP 562 `__getattr__`), with the eager-import consumers unchanged
- [x] A guard pins the leaf-import closure so the stack cannot creep back

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#1: met - Import census in this audit: importing tldw_chatbook.Library now pulls 25 tldw_chatbook modules (was 66) - the collections/sync service stack is behind the lazy facade.
- AC#2: met - PEP 562 __getattr__ facade at Library/__init__.py:43-48 resolves public names; eager surface is 3 leaf modules.
- AC#3: met - Guard shipped: Tests/Packaging/test_collections_capture_import_closure.py (116 lines) pins the leaf-import closure.
- Commit c64b781a74 'perf(collections): defer capture runtime past first paint' (2026-09-01) added the facade and the closure guard.
