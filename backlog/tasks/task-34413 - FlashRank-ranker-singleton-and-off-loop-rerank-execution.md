---
id: TASK-34413
title: FlashRank ranker singleton and off-loop rerank execution
status: Done
assignee: []
created_date: 2026-10-07 02:40
updated_date: 2026-10-07 03:16
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented module-level FlashRank ranker singleton (double-checked locking around construction only, failed construction retried not negative-cached) with cache dir at get_user_data_dir()/cache/flashrank; rerank inference moved off the Textual event loop via asyncio.to_thread (async _execute_process_step in pipeline_builder_simple with all call sites updated); tests use a _RANKER_FACTORY seam so flashrank stays optional. Measurement (real flashrank 0.2.10): pre-fix 115.1 ms per query on the event loop; post-fix warm hit 0.0006 ms, inference 25-35 ms in worker thread; first-ever cold 5010.8 ms (one-time download), subsequent fresh-process cold 211.4 ms. Pre-existing bug surfaced and parked as follow-up: passages lack id and ranked.index validation fails against real flashrank dicts so rerank silently falls back to unranked order. Files: RAG_Search/pipeline_functions_simple.py, RAG_Search/pipeline_builder_simple.py, Tests/RAG_Search/test_flashrank_ranker_reuse.py, Tests/RAG/test_local_citation_capture.py. Full report: .superpowers/sdd/2026-10-06-nonconsole-efficiency-remediation/task-1-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
