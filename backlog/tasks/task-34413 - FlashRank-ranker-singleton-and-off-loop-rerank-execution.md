---
id: TASK-34413
title: FlashRank ranker singleton and off-loop rerank execution
status: Done
assignee: []
created_date: '2026-10-07 02:40'
updated_date: '2026-10-07 03:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 1 / F1: the default-on rerank step rebuilds a torch Ranker per query and runs it synchronously on the Textual event loop freezing the UI on every RAG chat send
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ranker constructed once per process,Rerank and model load never run on the event loop,Ranker cache dir moved off /tmp to app cache dir,Warm vs cold measurement recorded in notes,Targeted RAG pipeline tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 1 (T1)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approach: lazy thread-safe process-wide ranker singleton
  (`get_flashrank_ranker()`) in `pipeline_functions_simple.py`; the lock
  guards construction only, so concurrent reranks share the instance
  without serializing. `_RANKER_FACTORY` is the test seam (keeps
  flashrank an optional, lazily-imported dependency).
  `_reset_flashrank_ranker_for_tests()` drops the cache for suites that
  swap the backend.
- Off-loop: `rerank_results` stays sync; new `rerank_results_async`
  wraps it in `asyncio.to_thread`. The builder's `_execute_process_step`
  is now `async def` and dispatches the rerank call through
  `asyncio.to_thread` (via the `PROCESSING_FUNCTIONS` registry entry, so
  monkeypatched fakes keep working); its only call site
  (`execute_pipeline`) awaits it.
- Cache dir: `/tmp` -> `get_user_data_dir()/cache/flashrank` (same home
  as the model-catalog cache; `tempfile.gettempdir()` only as fallback),
  so model weights survive a reboot.
- Measurement (real flashrank 0.2.10, this venv, M-series): first-ever
  construction incl. model download 5010.8 ms; fresh-process cold with
  weights persisted 211.4 ms; warm singleton hit 0.0006 ms; pre-fix
  fresh-Ranker-per-query cost 115.1 ms (in-process) -- all of it
  previously on the Textual event loop, per query. Real inference ~25-35
  ms per 10 passages, now off-loop (verified `rerank thread is not main
  thread` against the real ranker).
- Partial-failure semantics preserved: validation/fallback path and its
  score/provenance contract untouched. Existing fake-flashrank tests
  (`Tests/RAG/test_local_citation_capture.py`) updated only to reset the
  singleton when they swap the backend (`_install_flashrank` + the
  import-failure parametrization).
- Tests: new `Tests/RAG_Search/test_flashrank_ranker_reuse.py` (7
  tests: constructed-once, cache-dir off-/tmp, reset helper, off-loop
  via wrapper, off-loop via process step, end-to-end pipeline). RED then
  GREEN (AttributeError `_RANKER_FACTORY` -> 7 passed).
  `Tests/RAG/test_local_citation_capture.py` 103 passed. The
  `Tests/RAG_Search -k "pipeline or rerank"` selection shows the same 52
  pre-existing failures with and without this change (LLM-reranker /
  prompts-DB environment issues, unrelated).
- Known pre-existing issue surfaced by the real-path smoke test (NOT
  fixed here, out of scope): `rerank_results` builds passages without an
  `id` and validates `ranked.index` (attribute), but flashrank 0.2.10
  returns the input passage dicts with `score` added -- so every REAL
  flashrank rerank hits the `except Exception` fallback and returns
  original order. Rerank output has likely never been used in
  production; needs its own task.
- ADR required: no -- performance bugfix (caching + threading), no
  storage/sync/provider/security/UX-structure decision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
