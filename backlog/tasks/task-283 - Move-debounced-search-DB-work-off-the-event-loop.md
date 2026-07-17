---
id: TASK-283
title: Move debounced search DB work off the event loop
status: Done
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
- [x] #1 The three search paths execute their DB work off the event loop (asserted via a loop-blocking probe or thread-name check in tests)
- [x] #2 Stale results are still discarded via the existing cancellation tokens
- [x] #3 Existing search tests green
<!-- AC:END -->

## Implementation Plan

1. Re-verify line anchors against current code. Console browser search's actual leaf is two-fold: `ChatConversationScopeService.list_conversations` (chat_conversation_scope_service.py) for the scope-serviced candidate, AND a second, direct raw-`ChatConversationService.list_conversations` call in `ChatScreen._persisted_console_browser_rows` (chat_screen.py) for the `scope_service.local_service`/`app.local_chat_conversation_service` candidates in the same loop -- both needed threading, not just the scope service.
2. Thread `ChatConversationScopeService.list_conversations`'s local-mode call via `asyncio.to_thread`, gated on the local service's `.db.is_memory_db`; leave server mode (already a real coroutine) untouched.
3. Thread the raw-service branch in `_persisted_console_browser_rows` with the same guard.
4. Extract CCP conversation search's filter cascade (`perform_ccp_conversation_search`, conv_char_events.py) into a synchronous leaf function and thread it, gated on `db.is_memory_db`.
5. Thread media search's single `search_media_db` call (`perform_media_search_and_display`, media_events.py), same guard.
6. Console browser search already had a cancellation token (`_console_conversation_browser_search_token`, re-checked after the await in `_refresh_console_conversation_browser_search`) -- preserve it. CCP and media search had NO staleness guard at all; since threading widens the window where an older search's thread call can finish after a newer one starts (a debounce timer replacing a *pending* timer can't cancel an *in-flight* thread call), add a minimal per-search (CCP) / per-type_slug (media) generation counter and discard stale results after the threaded call, mirroring the console pattern.
7. Add new tests; run the existing search-related suites to confirm no regressions.

## Implementation Notes

**Console browser search**: `ChatConversationScopeService.list_conversations` (`tldw_chatbook/Chat/chat_conversation_scope_service.py`) now threads the local-mode call via `asyncio.to_thread` unless `_is_memory_backed(service)` (checks `service.db.is_memory_db`) or the target is already a coroutine function (server mode). `ChatScreen._persisted_console_browser_rows` (`tldw_chatbook/UI/Screens/chat_screen.py`) independently threads its raw-service branch (`include_mode=False`, i.e. `scope_service.local_service` / `app.local_chat_conversation_service`) with the same guard -- this branch bypasses the scope service entirely and was previously unthreaded even after fixing the scope service alone. The existing `_console_conversation_browser_search_token` staleness guard in `_refresh_console_conversation_browser_search` (checked after `await self._persisted_console_browser_rows(...)`) is untouched and still discards results from a search superseded while its thread call was in flight.

**CCP search**: `perform_ccp_conversation_search`'s filter cascade (title/keyword/tag/character-chat filters, 7 sync DB calls) is extracted into a new module-level leaf `_compute_ccp_conversation_search_results` (`tldw_chatbook/Event_Handlers/conv_char_events.py`) and run via `asyncio.to_thread` unless `db.is_memory_db`. Added a new `TldwCli._ccp_conversation_search_generation` counter (app.py) since this path had no staleness guard at all before; a search whose generation is superseded while off-thread is discarded before touching the results ListView.

**Media search**: `perform_media_search_and_display`'s single `search_media_db` call (`tldw_chatbook/Event_Handlers/media_events.py`) is threaded the same way, gated on `app.media_db.is_memory_db`. Added `TldwCli._media_search_generation` (a per-`type_slug` dict, since different media-type tabs search independently) for the same previously-absent staleness protection.

**Pre-existing bug found, NOT fixed (out of scope for this task)**: `perform_media_search_and_display`'s `from ..UI.MediaWindow import MediaWindow` (inside a `try/except QueryError`) always raises `ModuleNotFoundError` on current dev -- `tldw_chatbook/UI/MediaWindow.py` was removed and replaced by `MediaWindow_v2.py`/`MediaWindowV88.py` in an earlier, unrelated commit, and `ModuleNotFoundError` is not `QueryError` so it isn't caught there; it falls through to the function's outer broad `except Exception`, meaning **media search currently always renders "Error loading: No module named 'tldw_chatbook.UI.MediaWindow'" instead of real results**. This predates task-283 and is unrelated to the DB-threading fix; flagging it here (and in the batch report) rather than fixing it under this task's remit. `Tests/Chat/test_search_off_loop.py`'s media tests stub `sys.modules["tldw_chatbook.UI.MediaWindow"]` to exercise the threaded `search_media_db` path despite this.

**Files changed**: `tldw_chatbook/Chat/chat_conversation_scope_service.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/Event_Handlers/conv_char_events.py`, `tldw_chatbook/Event_Handlers/media_events.py`, `tldw_chatbook/app.py` (two new class-attribute generation counters). New test file: `Tests/Chat/test_search_off_loop.py` (12 tests: thread-affinity assertions for all three paths' file-backed vs. `:memory:`-backed DBs, plus stale-result discard for console/CCP/media). `Tests/Chat/` (917 passed, 69 skipped) and the targeted Console-browser/scope-service suites all green.
