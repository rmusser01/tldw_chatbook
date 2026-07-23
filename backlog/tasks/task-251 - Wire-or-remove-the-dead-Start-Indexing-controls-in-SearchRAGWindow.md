---
id: TASK-251
title: Wire or remove the dead Start Indexing controls in SearchRAGWindow
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:11'
labels:
  - rag
  - ux
dependencies:
  - TASK-247
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The standalone Search window (UI/Views/RAGSearch/search_rag_window.py) renders Index New Content controls whose start-indexing button (line 474) has no event handler anywhere in the codebase, along with index-stats controls that never update. The UI advertises indexing that cannot happen. Once a real bulk-index path exists (task-247) these controls should trigger it and reflect real index stats; otherwise they must be removed. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Search window contains no controls that do nothing when activated
- [x] #2 If indexing controls remain they trigger real indexing and display actual index statistics
<!-- AC:END -->

## Implementation Plan

1. Audit every control in SearchRAGWindow (compose + all @on handlers in the
   RAGSearch package) and record a wire-vs-remove disposition per dead control.
2. Wire #start-indexing to the real bulk-index path from task-247
   (`backfill_semantic_index`) on a thread worker (`@work(thread=True,
   exclusive=True, group=...)` + in-flight guard) with `asyncio.run`, exactly
   like the CLI entry point; honor #index-source-select via `item_types`;
   surface per-batch progress through the existing #indexing-status text via
   `call_from_thread`; catch all exceptions inside the worker.
3. Deps gate: when `embeddings_rag` is unavailable, disable the indexing
   controls in on_mount with the shared optional-dependency recovery tooltip
   (same copy as the disabled Search button); honest notify on the backfill
   summary (`unavailable` / failures with last error / indexed+skipped counts),
   re-enable the button and refresh stats on completion.
4. Make #index-stats-table real: read
   `vector_store.get_collection_stats()` from an ALREADY-initialized runtime
   only (`app._rag_service` or a new non-creating
   `peek_shared_rag_service()` accessor — never construct the embedding
   runtime just to render stats), with the trustworthy-count rule factored
   into `semantic_availability.trustworthy_collection_count()` and reused by
   `semantic_index_is_empty`. Honest "not initialized" / error rows otherwise.
5. Remove dead controls that have no cheap real backing: #create-collection,
   #delete-collection, #refresh-results, and the fake #indexing-progress bar.
6. Wire the remaining dead-but-cheap controls: #export-results -> existing
   export action, #prev-page/#next-page -> real pagination,
   #refresh-history + #history-range-select -> history reload with
   `days_back`, #refresh-collections -> existing collections+stats refresh
   workers.
7. TDD: failing tests first in Tests/UI/test_search_rag_window.py (deps-missing
   disable+tooltip, click -> backfill via worker with selected types,
   double-start guard, completion -> stats refresh, failure/unavailable ->
   honest notify, removed-controls assertions, pagination/history/export
   wiring); then run the full window suite + RAG ingestion/honest-state suites.

## Implementation Notes

Every rendered control in the standalone Search window now routes to a real
implementation or was removed. Per-control disposition (all were verified
dead: no `@on` selector or name-based handler anywhere in the codebase):

WIRED
- `#start-indexing` -> real bulk indexing via task-247's
  `backfill_semantic_index` on a thread worker (`@work(thread=True,
  exclusive=True, group="rag-index-backfill", exit_on_error=False)`) running
  `asyncio.run` exactly like the CLI entry point, with `_indexing_in_flight`
  single-flight guard, button disabled during the run, all exceptions caught
  inside the worker, and honest completion notifies (unavailable-with-install
  copy / failures with last error / indexed+up-to-date counts).
- `#index-source-select` -> read at press and mapped onto the backfill
  `item_types` contract (media / conversations / notes / all).
- `#index-stats-table` (previously populated with hard-coded zeros behind an
  always-None `self.rag_service` early-return) -> real
  `vector_store.get_collection_stats()` numbers read off-thread from an
  ALREADY-initialized runtime (`app._rag_service` or the new non-creating
  `peek_shared_rag_service()`); honest "not initialized" / error /
  untrustworthy-count rows otherwise; columns reduced to
  Collection/Chunks/Status (Documents/Size/Last-Updated had no real source).
- `#indexing-status` + `#indexing-status-text` -> shown during a run with
  real per-batch indexed/up-to-date/failed counts via the backfill
  progress callback marshalled through `call_from_thread`.
- `#refresh-collections` (relabeled "Refresh") -> existing collections-list
  worker plus the stats refresh.
- `#export-results` -> the existing Ctrl+E export action.
- `#prev-page` / `#next-page` -> real result pagination (pages beyond 1 were
  previously unreachable by mouse).
- `#refresh-history` + `#history-range-select` -> history-table reload
  honoring the selected range via `get_search_history(days_back=...)`.

REMOVED
- `#create-collection` / `#delete-collection`: no cheap real backing -- the
  simplified RAG service manages exactly one collection per profile.
- `#refresh-results`: re-running the query is exactly the Search button.
- `#indexing-progress` ProgressBar: the backfill total is unknown up front,
  so any percentage would be fake; real counts go to the status text.

ROOT-CAUSE FIX (made the whole window's mixin handlers real): none of
`SearchEventHandlersMixin`'s `@on` handlers -- including the Search button's
`handle_search` -- were EVER dispatched, because `@on` registration happens in
Textual's `_MessagePumpMeta` at class creation and dispatch walks each MRO
class's own `_decorated_handlers`; a plain mixin class never got either. The
mixin now uses `metaclass=_MessagePumpMeta` (identical to `Container`'s, so
no metaclass conflict). Also fixed the task-228-class hazard
`run_worker(exclusive=True)` without `group=` in the collections-apply seam
(it cancelled unrelated default-group workers, observable as
`WorkerCancelled` in tests and able to kill an in-flight search).

Deps gate: with `embeddings_rag` missing, `#start-indexing` and
`#index-source-select` are disabled in on_mount with the same shared
optional-dependency recovery tooltip as the Search button; the press path
re-checks and notifies with install copy, and `backfill_semantic_index`
itself re-gates via `semantic_indexing_available()`.

Left as-is (not activation no-ops): `#collection-select` / `#temperature-input`
still feed saved-search config only (search-flow scope stabilized by PR #692),
and `#search-progress` is a passive indicator in the search status row.

Files: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`,
`tldw_chatbook/UI/Views/RAGSearch/search_event_handlers.py`,
`tldw_chatbook/RAG_Search/ingestion_indexing.py` (new
`peek_shared_rag_service`), `tldw_chatbook/RAG_Search/semantic_availability.py`
(new `trustworthy_collection_count`, reused by `semantic_index_is_empty`),
plus tests in `Tests/UI/test_search_rag_window.py` (14 new),
`Tests/RAG/test_ingestion_indexing.py`, `Tests/RAG/test_semantic_honest_states.py`.

Verification: Tests/UI/test_search_rag_window.py 37 passed (23 baseline + 14
new), Tests/RAG/test_ingestion_indexing.py + test_semantic_honest_states.py
73 passed, combined 110 passed; `python -c "import tldw_chatbook.app"` OK.
