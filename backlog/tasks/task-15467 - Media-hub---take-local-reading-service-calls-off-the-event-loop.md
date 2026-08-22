---
id: TASK-15467
title: 'Media hub: take local reading-service calls off the event loop'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - media
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/MediaWindow_v2.py:2387` (`_run_media_search`) and `:1188-1191` (`_load_media_item_detail`) use `run_worker(coroutine)` — which does NOT leave the event loop — around plain synchronous `db.search_media_db(...)` (FTS + row enrichment) and `db.get_media_by_id(...)`; an item click additionally runs reading-progress, document-versions, and highlights queries, 3-4 sequential sync queries on the loop per click (`Media/media_reading_scope_service.py:628-687` `_maybe_await` -> sync `Media/local_media_reading_service.py:174-255`). The page-correction path doubles the search query. Every media search, keyword filter, pagination, subview change, undelete, and list-item click blocks input.

Fix direction: thread the local mode in the scope service, mirroring `ChatConversationScopeService`'s task-283 threading — including that task's lesson: use a POSITIVE-confirmation predicate ("confirmed file-backed -> thread"), not a negative one, so unrecognized service shapes/test doubles do not silently run inline. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No synchronous DB work runs on the event loop for media search, browse, pagination, or item click (evidence)
- [x] #2 Results, pagination, undelete, and detail loads identical (existing tests green)
- [x] #3 Item-click latency before/after on a large media DB recorded
<!-- AC:END -->

## Implementation Plan

1. Read the template (`Chat/chat_conversation_scope_service.py`, task-283): a
   positive-confirmation predicate -- `mode == LOCAL and not
   iscoroutinefunction(fn) and not is_memory_backed(service)` -- gates
   `asyncio.to_thread`; everything else (server mode, an already-async
   method, or a positively-confirmed `:memory:` DB) keeps running inline via
   the existing `_maybe_await`.
2. Sweep `Media/media_reading_scope_service.py` for every LOCAL-mode leaf
   call `UI/MediaWindow_v2.py` (the Media hub) actually invokes, and
   classify each as sync-passthrough (thread it) vs. already-async /
   server-only (leave untouched). Add one shared async helper
   (`_call_local_leaf`) implementing the predicate once, keyed on
   `service.media_db.is_memory_db` (the local attribute name; the chat
   template uses `.db`), and re-point every classified call through it.
3. For the item-click chain (`get_media_detail` + nested reading-progress +
   highlights + document-versions), decide thread-individually vs. batch
   into one `to_thread` hop; record the decision and why.
4. Write evidence: thread-affinity tests (file-backed threads off-loop,
   `:memory:`-backed and server-mode stay inline) for every converted leaf,
   plus a real-`MediaDatabase` + `sqlite3.Connection.set_trace_callback`
   test (the task-15463 evidence pattern) proving zero SQL statements
   execute on the calling/event-loop thread for the search leaf and the
   item-click chain. Mutation-test at least one guard by reverting it and
   confirming the test fails.
5. Run the existing Media suites unmodified first (must be green before
   touching anything), then the new evidence suite.
6. Record an honest before/after item-click latency measurement against a
   large (6,000-row) local media DB in an isolated scratch HOME.

## Implementation Notes

**Approach.** Added `MediaReadingScopeService._is_memory_backed` (mirrors
`ChatConversationScopeService._is_memory_backed`, keyed on `service.media_db`
-- the local service's actual attribute name, not `.db`) and
`_call_local_leaf(mode, service, method_name, *args, **kwargs)`, the shared
positive-confirmation gate: `fn = getattr(service, method_name)` (so a
service shape that lacks the method still raises the identical
`AttributeError` it always did, at the same point, before any threading
decision is made -- see the highlights caveat below), then thread via
`asyncio.to_thread` only when `mode == LOCAL and not
iscoroutinefunction(fn) and not is_memory_backed(service)`; otherwise fall
through to the pre-existing `_maybe_await(fn(*args, **kwargs))`. Every other
existing behavior (policy enforcement, normalization, error handling) is
untouched -- only the leaf DB call itself moved.

**Sweep result (every LOCAL-mode leaf `MediaWindow_v2.py` actually calls,
now threaded):** `search_media` (search/browse/pagination/keyword-filter,
and its page-correction retry, which reuses the same method), `list_read_it_
later` (the read-it-later browse subview -- same underlying
`local_service.search_media` leaf), `get_media_detail` (item click; also
threads its own internal nested `get_reading_progress` call),
`get_reading_progress` (standalone), `list_reading_highlights`,
`list_document_versions`, `delete_media`, `undelete_media`,
`create_reading_highlight`, `update_reading_highlight`,
`delete_reading_highlight`, `save_analysis_version`,
`overwrite_analysis_version`, `delete_analysis_version`,
`save_to_read_it_later`, `remove_from_read_it_later`,
`update_media_metadata` (the leaf `update_media_metadata_latest`'s
last-write-wins lock ultimately calls). Server mode and every other scope-
service method (ingest/process operations, saved searches, note links,
digest schedules, `list_unified_items`/`get_unified_item` and the
`list_backing_media_*`/`get_backing_media_item` family the Library screen
uses) were left untouched: they are either already async, server-only, or
-- for the Library-only family -- not reachable from the Media hub at all
(Library screen already threads its own scope-service calls at the caller
level via `_run_library_service_call(..., isolate_in_worker=True)`, an
independent, coarser-grained mechanism; the 3 methods it shares with the
Media hub, `save_analysis_version`/`save_to_read_it_later`/`remove_from_
read_it_later`, now get a harmless extra thread-hop when Library calls them
-- functionally identical, one more context switch).

**Item-click batching decision.** Threaded individually (detail, then
highlights, then document-versions -- each its own `to_thread` hop) rather
than bundling into one combined leaf call. Reasons: (1) it preserves the
existing per-stage `_detail_presentation_is_current` staleness re-check
between each `await`, so a selection change mid-load still bails without
doing the remaining work, exactly as before; (2) each hop is a single
indexed-PK sqlite read (confirmed cheap by the latency probe below, <1ms
each), so the extra ~0.1-0.3ms per hop is imperceptible next to the
compounding effects the audit measured elsewhere (CSS reparse, screen
switch); (3) a bundled call would need a brand-new scope-service method and
its own test surface for a case where the underlying queries are already
fast -- not worth the risk for this task's scope.

**Pre-existing bugs found, NOT fixed (out of scope, flagged per the
task-283 convention):**
- `LocalMediaReadingService` never implemented the `reading_`-prefixed
  highlight CRUD the scope service calls in local mode -- all **four**
  methods are affected, not just the one first spotted: `list_reading_
  highlights`, `create_reading_highlight`, `update_reading_highlight`, and
  `delete_reading_highlight` all `AttributeError` against a real local
  service, because `LocalMediaReadingService` only implements the
  unprefixed `list_highlights`/`create_highlight`/`update_highlight`/
  `delete_highlight` (the Library screen's own, separate call sites use
  those correct, unprefixed names and work fine). Every local-mode item
  click already hit `AttributeError` loading highlights before this change
  (caught by `_load_media_item_detail`'s broad `except Exception`,
  presented as zero highlights), and every local-mode highlight
  create/update/delete action already hit the identical `AttributeError`
  too. Confirmed all four directly against a real `LocalMediaReadingService`
  instance (`getattr(service, "create_reading_highlight")` etc. all raise).
  `_call_local_leaf`'s `getattr(service, method_name)` raises the identical
  `AttributeError` at the identical point, before any threading decision is
  made, so this task changes nothing about the surface, thread, or timing of
  that failure for any of the four -- it was already broken, still is,
  unrelated to the event loop.
- `MediaWindow_v2.py:1685` (`handle_analysis_request`'s `perform_analysis`,
  dispatched via `run_worker(coroutine)`) has its own, separate direct
  `self.app_instance.media_db.get_media_by_id(...)` call, bypassing the
  scope service entirely -- a rare fallback branch (only when the event's
  own record is empty), structurally the same "run_worker(coroutine) != a
  thread" bug the audit named, but not one of the two audited sites and not
  on the search/browse/item-click/undelete AC's path. Left unthreaded; flag
  for a follow-up if it turns out to matter in practice.

**Evidence (AC#1).** New `Tests/Media/test_media_reading_scope_service_off_
loop.py` (40 tests): a parametrized thread-affinity double covering all 17
converted call sites, run twice -- file-backed (must NOT run on the caller
thread) and `:memory:`-backed (must stay on the caller thread, the
task-283 hazard); a test proving an unrecognized local double with no
`media_db` still threads (positive-confirmation predicate, no negative
branch); a test proving server mode never threads even for a sync double
(the gate is `mode == LOCAL`, not merely "is this sync?"); and two real-
`MediaDatabase` + `sqlite3.Connection.set_trace_callback` tests (the
task-15463 evidence pattern) asserting zero SQL statements land on the
calling/event-loop thread's own connection for the search leaf and for the
item-click chain (`get_media_detail` + its internal reading-progress fetch,
plus `list_document_versions`; `list_reading_highlights` deliberately
excluded from this real-DB test per the pre-existing-bug note above).
Mutation-tested by reverting the `search_media` conversion by hand: both the
thread-affinity test and the trace-callback test failed with the exact
statements/thread-identity evidence in the assertion message, then restored
and re-confirmed green.

**AC#2 (existing tests green, unmodified first).** CORRECTION (review round,
2026-08-11): the counts originally recorded here were wrong -- each file was
run individually with `pytest <file> -q` and its own "N passed" line read
directly, replacing an earlier accounting error where two files run together
in one command had their combined count misattributed to a single file.
Verified individually: `Tests/Media/test_media_reading_scope_service.py`
(**73**), `Tests/Media/test_local_media_reading_service.py` (**69**, not
100), `Tests/UI/test_media_window_v2_parity.py` (**31**, not 78) --
**173 pre-existing tests**, all green before any code change and still green
after. Adding the new `Tests/Media/test_media_reading_scope_service_off_
loop.py` (**40**, new -- not a pre-existing-regression check) brings the
combined targeted total to **213** when all four files are run together
(`pytest Tests/Media/test_media_reading_scope_service.py Tests/Media/test_
local_media_reading_service.py Tests/UI/test_media_window_v2_parity.py
Tests/Media/test_media_reading_scope_service_off_loop.py -q`), not the 291
previously claimed. These tests mostly use
`Mock`/`AsyncMock` scope services with `runtime_backend="server"` (never
threaded) or `FakeLocalMediaService` doubles with no `media_db` attribute
(now threaded, but the tests only assert on the returned value / recorded
call args, which `await asyncio.to_thread(...)` preserves exactly since the
`await` blocks until the thread finishes). One real-DB test in the existing
suite (`test_scope_service_local_detail_carries_saved_state_from_local_
service`) already used a `:memory:` `Database` + real
`LocalMediaReadingService` and stayed green, confirming the memory-backed
guard degrades correctly for that pre-existing case too. Also ran the wider
consumer surface (`Tests/Library/`, `Tests/RuntimePolicy/test_unsupported_
capabilities.py`, `Tests/MCP/test_library_tools.py`,
`Tests/ProductionApp/test_media_state_ownership.py`,
`Tests/ProductionApp/test_service_composition_lifecycle.py`,
`Tests/UI/test_home_screen.py`): identical failure SET before and after (13
failed / 15 errors in the two `ProductionApp` files, confirmed pre-existing
by swapping in the pre-change file and re-running the identical command --
same 13 + 15 test names both times; root cause looks environmental, a
blocked real network-egress attempt plus a screen-routing timeout, unrelated
to this change). Everything else in that sweep (1789 in `Tests/Library/`
alone) passed both before and after.

**AC#3 (latency probe, isolated scratch HOME).** Script at
`/private/tmp/.../scratchpad/t15467_latency_probe.py` (session-scoped
scratch, not committed): `HOME`/`XDG_CONFIG_HOME`/`XDG_DATA_HOME`/
`TLDW_CONFIG_PATH` all pointed at a fresh `tempfile.mkdtemp()` before any
`tldw_chatbook` import (never touches the real profile). Built a 6,000-row
local media DB (~2.4 KB article body each, distinct per row -- the DB
dedups by content, not just title, which the first probe attempt found the
hard way) and measured both raw wall time and event-loop starvation (a 1ms
heartbeat coroutine racing the DB call; missed ticks == time the loop
couldn't run anything else) for the search leaf and the item-click chain,
8 repeats each:

| leaf | wall(ms) median | heartbeat ticks median |
|---|---|---|
| search, BEFORE (inline) | 1059 | 0 |
| search, AFTER (threaded) | 982 | 831 |
| item-click chain, BEFORE (inline) | 0.10 | 0 |
| item-click chain, AFTER (threaded) | 0.39 | 0 |

Honest reading: search on this dataset is genuinely slow (~1s) because the
FTS fallback's `LIKE '%query%'` clause over the `content` column can't use
an index and full-scans ~14 MB of text -- that is exactly the shape a real,
sizeable local library search hits, and the AFTER row proves the loop stays
almost perfectly live throughout (831 ticks at 1ms cadence over a ~980ms
call) where BEFORE it is completely frozen for the full second. The
item-click chain is fast regardless on this dataset (indexed
primary-key/foreign-key reads) -- both numbers are sub-millisecond, so no
starvation is visible at this scale either way, and AFTER is measurably
*slower* in raw wall time by the ~0.1-0.3ms/hop `asyncio.to_thread` context-
switch cost noted in the batching decision above. Not fixed by batching (see
that decision); the win from threading the item-click chain is correctness
under slower disks/hardware (the audit's 3-5x constrained-hardware
multiplier) and consistency with search, not a measured latency win on this
fast local SSD with a small per-item DB.

**Files changed:** `tldw_chatbook/Media/media_reading_scope_service.py`
(the helper + 17 converted call sites). New test file:
`Tests/Media/test_media_reading_scope_service_off_loop.py` (40 tests).
