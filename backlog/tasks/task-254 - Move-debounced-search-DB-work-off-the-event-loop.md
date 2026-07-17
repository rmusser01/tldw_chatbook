---
id: TASK-254
title: Move debounced search DB work off the event loop
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, chat]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
run_worker(coroutine) is NOT a thread: Console browser search (chat_screen 661-671 → chat_conversation_scope_service._maybe_await → SYNC list_conversations ×(1+workspaces)), CCP search (conv_char_events 615/1248), and media search (media_events 363/518) all execute sync sqlite/FTS on the event loop when their debounce fires (~3ms p50 @1.5k convs, grows with data). Fix at the service leaf with asyncio.to_thread; connections are thread-local. Note exclusive=True cancellation cannot interrupt an in-flight thread call — guard result application with the existing tokens. NOTE (review): sqlite :memory: databases are per-connection, so unconditional asyncio.to_thread at the leaf would hand test DBs a separate EMPTY database — gate the offload on `not db.is_memory_db` (the attribute already exists on CharactersRAGDB). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three search paths execute their DB work off the event loop (asserted via a loop-blocking probe or thread-name check in tests)
- [ ] #2 Stale results are still discarded via the existing cancellation tokens
- [ ] #3 Existing search tests green
<!-- AC:END -->
