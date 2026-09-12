---
id: TASK-15471
title: 'Event-loop I/O sundries: per-click writes and lookups off the loop'
status: Done
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch of small, individually-verified per-click blockers (July task-261 precedent), from the audit: Console conversation star toggle runs a sync read + write transaction (fsync) on the loop (`UI/Console_Modules/workspace.py:1946-2013`; browser refresh also reads starred ids sync at `chat_screen.py:9914`); Study create-card/add-topic write ChaChaNotes synchronously in handlers (`Event_Handlers/Study_Events/study_events.py:46-138`) and the Study dashboard fallback queries run on resume; emoji picker rewrites its recents JSON per selection (`Widgets/emoji_picker.py:74-113`); TTS export does `shutil.copy2` (MBs) + `json.dump` on the loop (`Event_Handlers/TTS_Events/tts_events.py:2510-2550`); collections keyword delete runs 2xN redundant SELECTs on the loop before its properly-threaded delete (`Event_Handlers/collections_tag_events.py:142-189`); `chat_message_enhanced.handle_save_image` writes multi-MB images sync (`:634-650`; the Console-side equivalent in `UI/Console_Modules/message.py:1501-1626` is already threaded — copy it); the enhanced file picker's search stats every directory entry twice per keystroke with no debounce (`Widgets/enhanced_file_picker.py:650/:686/:708-715`); CodeRepoCopyPaste reads whole files per tree-node click (`UI/CodeRepoCopyPasteWindow.py:716/:901`); ChatbookExportManagement globs + reads manifests per refresh (`:496`).

Fix direction: to_thread / debounce / dedupe per site, smallest-diff first; no behavioral changes. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed site is threaded, debounced, or deduped — or explicitly justified in the notes
- [x] #2 Behavior unchanged across the touched surfaces (targeted tests where they exist)
- [x] #3 Spot latency evidence recorded for the star toggle and file-picker typing
<!-- AC:END -->

## Implementation Plan

All sites re-located and re-verified at HEAD `c3ed2854a` (dev has moved ~300 commits
since the audit of `82b595049`; several cites shifted files/lines).

1. Star toggle (`UI/Console_Modules/workspace.py:_toggle_console_conversation_star`,
   now ~1970): keep the sync guards, move the marks-service `is_starred` +
   `star/unstar_conversation` (a read + a write transaction with fsync on
   chachanotes) into a `run_worker` async wrapper that calls
   `asyncio.to_thread(...)` with the repo's `is_memory_db` guard (pattern copied
   from `chat_screen.py` ~10550, task-15455's browser-search threading; the
   controller exposes `run_worker` as a live property). Serialize toggles with an
   `asyncio.Lock` so a rapid double-click still nets toggle-twice (deterministic,
   matching sync semantics) instead of racing two threads.
2. Browser starred-ids refresh (`chat_screen.py:_starred_console_conversation_ids`,
   now ~10305, called from ~7 refresh paths): dedupe at the source — add a
   result cache to `ConversationLocalMarksService.list_marked_conversation_ids`
   keyed by (mark_type, limit), invalidated by `set_mark`/`clear_mark` under a
   `threading.Lock` (service is now called from pool threads).
3. Study (`Event_Handlers/Study_Events/study_events.py`): justify-no-change — the
   audited handlers are dead code at HEAD (`STUDY_BUTTON_HANDLERS` /
   `study_event_handler` referenced nowhere outside the module; flashcard actions
   moved to `UI/Study_Modules/flashcards_handler.py` via the scope service).
   Study dashboard fallback queries (`UI/Screens/study_screen.py:
   _refresh_dashboard_snapshot` ~1088/1104/1123, direct sync `db.*` calls on
   resume): wrap in `asyncio.to_thread` with the `is_memory_db` guard.
4. Emoji picker (`Widgets/emoji_picker.py:1037/:1053`): hand `save_recent_emoji`
   (load+rewrite of recents JSON per selection) to an app-owned thread worker
   (`self.app.run_worker(..., thread=True, exit_on_error=False)`) — app-owned so
   the immediately-following `dismiss()` cannot cancel it.
5. TTS export (`Event_Handlers/TTS_Events/tts_events.py:handle_tts_export`
   ~2531): move mkdir + `shutil.copy2` + metadata `json.dump` into one
   `asyncio.to_thread` closure inside the existing try/except.
6. Collections keyword delete (`Event_Handlers/collections_tag_events.py:
   handle_keyword_delete`): dedupe 2×N SELECTs into one threaded batch lookup
   reused for both the notification names and the delete loop. Discovered while
   here: `app.run_in_thread` does not exist (Textual 8.2.8 App has no such
   method; nothing in the repo defines it), so every "threaded" call in this
   file raised AttributeError — rename/merge/delete were all broken. Replace all
   `app.run_in_thread` uses in this file with `asyncio.to_thread` (behavior
   repair, documented in notes).
7. Save image (`Widgets/Chat_Widgets/chat_message_enhanced.py:handle_save_image`
   ~635 — the audit's cite; `Widgets/chat_message_enhanced.py` is now an alias
   shim): `await asyncio.to_thread(...)` for the multi-MB `write_bytes`, copying
   the already-threaded Console pattern (`UI/Console_Modules/message.py:1645`).
8. File picker search (`Widgets/enhanced_file_picker.py`): (a) debounce — add a
   0.2 s debounced `_watch_search_filter` on `SearchableDirectoryNavigation`
   (same interval as the rail-search debounce reference, task-15454);
   (b) dedupe — merge `_repopulate_display`'s second full-directory
   `is_file`/`is_hidden` pass (the `filter_hidden` count, ~724) into the main
   entry loop so each entry is stat-ed once, not twice, per repopulate.
9. CodeRepoCopyPaste (`UI/CodeRepoCopyPasteWindow.py:716/:901`): thread the
   local-file preview read and the compile-selected read loop via
   `asyncio.to_thread`.
10. ChatbookExportManagement (`UI/ChatbookExportManagementWindow.py:
    refresh_chatbook_list` ~489 glob+stat scan; `_load_chatbook_details` ~966
    zip-manifest preview per selection): thread both via `asyncio.to_thread`;
    remove the two now-stale BASELINE entries in
    `Tests/Architecture/test_no_blocking_io_on_message_pump.py` (its
    stale-entry test goes red on the fix and green on the baseline update —
    born-red evidence).

Evidence plan (AC3): direct timing harnesses under scratch HOME/XDG/
TLDW_CONFIG_PATH with PYTHONPATH pinned to this worktree — (a) star toggle:
time the on-loop portion before (service read+write, file-backed scratch DB)
vs after (guards + worker spawn); (b) picker typing: 1000-file scratch dir,
count `is_file` calls and time `_repopulate_display` per keystroke before vs
after, plus debounce coalescing. Targeted tests for every touched surface;
failures baselined against origin/dev before attribution.

## Implementation Notes

All nine audit sites re-located at HEAD `c3ed2854a` and handled with the
smallest per-site diff; every claim below names its measurement or command.

**Per-site outcome**

1. **Star toggle — THREADED + DEDUPED.**
   `workspace.py:_toggle_console_conversation_star` keeps its sync guards and
   dispatches a `run_worker` async wrapper that runs the `is_starred` +
   `star/unstar` pair via `asyncio.to_thread` (with the `is_memory_db` guard
   copied from `chat_screen.py`'s task-15455 browser-search threading), then
   toasts + re-syncs. A new controller-level `asyncio.Lock` serializes
   presses so a rapid double-click still nets toggle-twice deterministically.
   The browser-refresh read (`_starred_console_conversation_ids`, ~7 call
   paths) is deduped at the source: `ConversationLocalMarksService.
   list_marked_conversation_ids` now caches per (mark_type, limit) under a
   `threading.Lock`, invalidated by `set_mark`/`clear_mark` — every star
   writer in the process goes through this instance.
2. **Study — JUSTIFIED (dead code) + fallbacks THREADED.** The audited
   handlers in `Event_Handlers/Study_Events/study_events.py` are unreachable
   at HEAD: `grep -rn "STUDY_BUTTON_HANDLERS\|study_event_handler"` finds no
   reference outside the module; flashcard actions moved to
   `UI/Study_Modules/flashcards_handler.py` via the scope service. (Aside:
   the `#add-topic-btn` in `StructuredLearningWidget` currently has NO
   handler at all — a pre-existing functional gap, out of this task's
   scope.) The Study dashboard's three sync fallback queries in
   `study_screen.py:_refresh_dashboard_snapshot` (`get_due_flashcards`,
   `list_decks`, `list_quizzes` on resume) now run via `asyncio.to_thread`
   with the `is_memory_db` guard.
3. **Emoji picker — THREADED.** `save_recent_emoji` (read+rewrite of the
   recents JSON per selection) is handed to an app-owned
   `run_worker(thread=True, exit_on_error=False)` — app-owned because the
   picker dismisses immediately and a screen-owned worker would be cancelled
   before running.
4. **TTS export — THREADED.** `handle_tts_export`'s mkdir + `shutil.copy2`
   + metadata `json.dump` moved into one `asyncio.to_thread` closure inside
   the existing try/except; toasts unchanged.
5. **Collections keyword delete — DEDUPED + THREADED (+ behavior repair,
   disclosed).** The 2×N per-id SELECTs collapsed to one threaded batch
   lookup reused by both the notification and the delete loop. Discovered
   while here: `app.run_in_thread` does not exist — Textual 8.2.8's `App`
   has no such method (`hasattr(App, "run_in_thread") == False`) and nothing
   in the repo defines it — so every "threaded" call in this file raised
   `AttributeError` into its own error toast (rename/merge/delete were all
   broken). All four call sites replaced with a small `_media_db_off_loop`
   helper (`asyncio.to_thread` + `is_memory_db` guard;
   `Client_Media_DB_v2` uses thread-local connections). This repairs the
   handlers — a deliberate, disclosed behavior fix, pinned by the new tests.
   **Follow-up flagged:** `Event_Handlers/multi_item_review_events.py` has
   the same nonexistent `app.run_in_thread` at 4 sites — NOT fixed here
   (uncited by the audit).
6. **Save image — THREADED.** `Widgets/Chat_Widgets/chat_message_enhanced.py:
   handle_save_image` (the audit's cite now lives here; the old path is an
   alias shim) awaits `asyncio.to_thread(write_bytes)` with the bytes
   snapshotted first — copying the already-threaded Console pattern
   (`UI/Console_Modules/message.py:_save_console_message_image`).
7. **File picker search — DEBOUNCED + DEDUPED.**
   `SearchableDirectoryNavigation` gains a 0.2 s debounced
   `_watch_search_filter` (same interval as the rail-search reference,
   task-15454; widget-owned timer). `_repopulate_display`'s second
   full-directory pass (the task-431 `filter_hidden` count) merged into the
   main loop: `is_file` now runs at most once per entry, and not at all
   when no `file_filter` is set (it previously ran unconditionally, first
   in the `and` chain). Predicates are the vendored `hide()` unrolled;
   the visible set and both posted counts are unchanged (pinned by
   `test_filter_hidden_count` — whose fixture the review fix round extended
   with a dotfile, because without one the `not dot_hidden` guard was
   unpinned (review minor 7) — plus the new debounce test).
8. **CodeRepoCopyPaste — THREADED.** The per-click local-file preview read
   (`:716`) and the compile-selected read loop (`:901`) both moved into
   `asyncio.to_thread` closures; per-file error handling preserved.
9. **ChatbookExportManagement — THREADED.** `refresh_chatbook_list`'s
   glob+stat scan extracted to `_scan_chatbook_files` run via
   `asyncio.to_thread` (list identity kept via clear+extend), and
   `_load_chatbook_details`' per-selection zip-manifest `preview_chatbook`
   threaded the same way. The two now-stale BASELINE entries in
   `Tests/Architecture/test_no_blocking_io_on_message_pump.py` were removed
   — its stale-entry test went red on the code fix alone (born-red
   evidence) and the suite is 6/6 green after the baseline update.

**AC3 latency evidence** (direct timing harnesses, scratch
HOME/XDG/TLDW_CONFIG_PATH, `tldw_chatbook.__file__` asserted to this
worktree; scripts `probe_star_toggle.py` / `probe_star_after.py` /
`probe_picker_search.py` in the session scratchpad):

- *Star toggle*, file-backed scratch chachanotes DB, n=60: on-loop work
  BEFORE (is_starred + star/unstar write) median **0.063 ms** (max 0.089);
  AFTER the on-loop portion is the worker dispatch, median **0.0025 ms**,
  with the durable write off-loop (end-to-end median 0.218 ms). The before
  median is small on this dev box with an empty DB — the tail hazard the
  threading removes is the sqlite `busy_timeout` block when another writer
  holds the file, which a quiet-machine median cannot show. Browser-refresh
  starred-ids read: **0.031 ms**/SELECT before → **0.0002 ms** cache hit
  after.
- *File-picker typing*, 1000-file scratch directory, mounted
  `SearchableDirectoryNavigation` under a Pilot: BEFORE, every keystroke ran
  one full repopulate at **121.1 ms median** on the loop (5-keystroke burst
  = 5 repopulates ≈ 605 ms blocked); `is_file` stat calls per repopulate:
  **1000**. AFTER, the same 5-keystroke burst = **1** repopulate (0.2 s
  after typing pauses) and **0** `is_file` calls with no filter set (N, not
  2N, with one). Single-repopulate wall time is unchanged (~134 ms,
  dominated by the option-list rebuild itself — a separate lever, not this
  task's).

**Tests** (all via the repo venv with PYTHONPATH pinned to this worktree):
Architecture no-blocking-io guard 6/6; picker suites
(`Tests/test_enhanced_filepicker.py` + 4 UI files) 46 passed + new
`test_search_keystrokes_debounce_into_one_filtered_repopulation`
(mutation-verified born-red: inline-rebuild mutation fails it); marks
service 10 passed + new
`test_list_marked_ids_cache_never_serves_stale_results_across_writes`
(mutation-verified: removing `clear_mark`'s invalidation fails it); new
`Tests/Event_Handlers/test_collections_tag_events.py` 2 passed (both red
under a `run_in_thread` restoration AND the delete test red under a
duplicated-lookup mutation); Console button routing **16 passed** and full
workspace-context-rail **69 passed** (corrected in the review fix round —
the earlier notes mis-tallied these from batch outputs as 22/96; the
per-file counts above are re-run and read directly) — the three star tests
now wait on `workers.wait_for_complete()` (the write moved to a worker;
assertions were racing the pool thread), stable across 3 repeat runs;
code-repo window + integration, chatbook management (9), TTS improvements
+ study product-maturity suites (45), chat-message widget + artifact
actions + state ownership (155 with full rail), focus-contract +
speak-autoplay (115).

**Failures attributed to dev, not this change** (each reproduced byte-for-
byte on a pristine `c3ed2854a` throwaway worktree, then the worktree
removed): `Tests/Chat/test_conversation_local_marks_service.py::
test_local_marks_migrate_from_v16_to_v17_with_expected_schema`
(v35→v36 migration: "table note_folders already exists") and
`Tests/UI/test_product_maturity_phase3_knowledge_entry.py::
test_study_screen_consumes_pending_initial_section` (network-egress
teardown guard: `socket.connect -> 104.18.x.x:443`). Neither is on
task-15766's list yet — candidates for that batch.

**Modified files:** `tldw_chatbook/Chat/conversation_local_marks_service.py`,
`UI/Console_Modules/workspace.py`, `UI/Screens/study_screen.py`,
`Widgets/emoji_picker.py`, `Event_Handlers/TTS_Events/tts_events.py`,
`Event_Handlers/collections_tag_events.py`,
`Widgets/Chat_Widgets/chat_message_enhanced.py`,
`Widgets/enhanced_file_picker.py`, `UI/CodeRepoCopyPasteWindow.py`,
`UI/ChatbookExportManagementWindow.py`; tests:
`Tests/Architecture/test_no_blocking_io_on_message_pump.py` (baseline),
`Tests/Chat/test_conversation_local_marks_service.py`,
`Tests/UI/test_enhanced_file_dialog_mount.py`,
`Tests/UI/test_console_button_routing.py`,
`Tests/UI/test_console_workspace_context_rail.py`,
`Tests/Event_Handlers/test_collections_tag_events.py` (new).
Ruff (corrected in the review fix round): `check` on the touched files
reports exactly **1 production finding** — the pre-existing F811 duplicate
`DEFAULT_WORKSPACE_ID` import in `workspace.py` (present at base
`c3ed2854a`) — plus **3 F401s in `Tests/UI/test_console_workspace_context_
rail.py`**, all three also present at base (verified by running
`ruff check --isolated --select F401` on the base blob: they report at base
lines 2610/2747/2768, shifted only by this branch's appended tests). The
earlier "4 pre-existing findings" lumped these without the
production-vs-test split. `format --check` clean on all files that were
clean at base; base-format-dirty files were not reformatted, and the one
new hunk the formatter flagged inside this branch's own additions (the
marks cache test's wrapped assert) was conformed by hand.

## Review fix round (independent concurrency review, verdict FIX-FIRST)

Review report: session scratchpad `review15471/review.md` (all its findings
reproduced by the reviewer; all reproduced again here before fixing).

- **M1 (blocking) — populate-after-invalidate cache race, FIXED.**
  `conversation_local_marks_service.py`: a cache-missing reader held its
  fetched rows across the transaction COMMIT and stored them without
  re-checking for a concurrent invalidation — a writer committing +
  invalidating in that window (star toggle pool thread, fleet drain child
  thread) left a pre-write snapshot cached until the NEXT mark write
  (just-starred shows unstarred; a FLEET_UNSEEN badge never appears). Fix
  is the reviewer's suggested shape: `_invalidate_list_cache` bumps a
  generation counter under the lock; the reader captures it in the same
  lock hold as its cache miss and stores only if unchanged — a lost race
  costs one skipped store, never a stale cache. Kept the cache (vs
  dropping it) because the guard is small and obviously correct: a global
  counter, capture-before-SELECT, compare-before-store.
  Evidence: new test `test_list_cache_is_not_repopulated_with_a_pre_write_
  snapshot` (the reviewer's deterministic interleave adapted — reader
  paused exactly at its commit-exit) was **born red against the committed
  code** (`assert () == ('conv-a',)`, the exact staleness) and is green
  after the fix; the reviewer's own probes re-run against the fix:
  `probe_cache_race.py` → "no staleness observed",
  `probe_cache_race_natural.py` → **stale rounds 0/300** (was 103/300).
- **M2 — TTS export zeroed-file window, FIXED.** The threaded `copy2`
  yields, so the 5 s `_cleanup_audio_file` task could secure-delete the
  source mid-copy — and that delete overwrites the file IN PLACE with
  zeros before unlinking (`Utils/secure_temp_files.py`), so the export
  silently wrote zeros under a success toast. Boring durable option
  chosen: the export **claims** the message id (refcounted dict guarded by
  the existing `_audio_files_lock`, claimed in the same lock hold that
  reads the source path, released in a `finally`), and
  `_cleanup_audio_file` polls the claim (0.25 s interval, loop not
  recursion) before deleting — copy and destroy are now mutually
  exclusive, cleanup is deferred (never skipped, so no leak), and
  refcounting keeps overlapping exports of one message safe. Evidence: new
  test `test_export_claim_keeps_cleanup_from_destroying_the_source_mid_
  copy` (gated `shutil.copy2`, cleanup fired mid-copy) — **born red
  against the committed code** ("cleanup destroyed the source mid-copy"),
  green after; it also pins that the deferred cleanup still deletes the
  source once the export finishes.
- **Minor 3 — star worker cancellation, FIXED.** `CancelledError` is a
  `BaseException` and sailed past the `except Exception`, recreating the
  TASK-357 silent-toggle shape (pool-thread write lands, no repaint). The
  wrapper now catches `asyncio.CancelledError`, best-effort re-syncs the
  workspace context, and re-raises.
- **Minor 4 — debounce deferred clears, FIXED.** An emptied
  `search_filter` (Esc / Clear button / programmatic reset) now
  repopulates immediately; only non-empty typing debounces. Pinned by an
  extension to the debounce test: after the debounced filter, `search_
  filter = ""` must repopulate synchronously (count increments on the
  assignment line) and restore the full listing.
- **Minor 5 — emoji recents ordering, DOCUMENTED.** The unserialised
  last-write-wins semantics (and why a lock is not worth it for a
  cosmetic, error-swallowing recents file) are now stated in
  `_save_recent_emoji_off_loop`'s docstring.
- **Minor 6 — chatbook list cleared before the await, FIXED.**
  `refresh_chatbook_list` now scans first and does clear+extend after the
  await returns, so a failed scan leaves the previous state instead of an
  empty list under a stale OptionList (the IndexError setup the review
  described).
- **Minor 7 — evidence honesty, CORRECTED IN PLACE.** The test-count and
  ruff numbers above now carry the re-run values (routing 16, rail 69;
  ruff 1 production + 3 test-file findings, all verified at base), and the
  previously-unpinned `not dot_hidden` guard is now genuinely pinned: the
  `test_filter_hidden_count` fixture gained a dotfile that also fails the
  filter (count must stay 3), verified born-red by a drop-the-guard
  mutation (count becomes 4) and restored.

Fix-round test evidence: marks suite 18 passed (+ the known dev-red
migration test, baselined on pristine base last round); TTS improvements
25 passed; picker sweep + collections + arch guard + chatbook management
64 passed; button routing 16 passed; workspace-context-rail 69 passed —
all after the fixes, `PYTHONPATH` pinned to this worktree.
