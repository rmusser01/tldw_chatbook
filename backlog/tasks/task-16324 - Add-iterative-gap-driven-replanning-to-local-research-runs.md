---
id: TASK-16324
title: Add iterative gap-driven replanning to local research runs
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:15'
updated_date: '2026-08-15 13:34'
labels:
  - research
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 generates sub-queries once and never revises them as evidence arrives. Add a bounded iteration loop modeled on tldw_server stop_criteria: after an initial synthesis pass, identify thin or unanswered sub-questions, generate follow-up sub-queries, and loop within budget and max_iterations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After an initial synthesis pass a gap-analysis step identifies unanswered or thin sub-questions
- [x] #2 Follow-up sub-queries are generated and executed bounded by max_iterations and the budget ledger
- [x] #3 The final report reflects evidence gathered across iterations and names remaining gaps
- [x] #4 Iteration transitions are visible in the run event stream
- [x] #5 Tests pin the iteration bound and gap reporting
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD engine iteration loop: injectable gap_fn (default returns empty without a configured synthesis LLM - never breaks the run), loop collect plus synthesize while gaps are returned and iteration < max_iterations from limits_json (default 1 preserves single-pass behavior), merge results across iterations with URL dedup, record iteration_started events, surface remaining gaps in report_v1.md and bundle.json
2. TDD budget interaction: a gap iteration that cannot reserve searches stops cleanly through the existing research_limit_exceeded path
3. Tests plus lint plus task close
ADR required: no - same engine contract (ADR-068) gaining a bounded loop through its existing seams; gap_fn mirrors the search_fn and analyze_fn injection pattern
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- The engine's linear collect-then-synthesize section became a bounded loop (task-16324): iteration 1 researches the question; every later round researches the gaps the previous synthesis left open. Results merge across rounds with URL dedup and each synthesis runs over the merged evidence. `max_iterations` comes from `limits_json` (default 1 — single-pass behavior is unchanged unless a run opts in, mirroring the server's `stop_criteria.max_iterations`).
- Gap analysis is an injectable `gap_fn(context)` seam matching the `search_fn`/`analyze_fn` pattern. The default uses the synthesis LLM (`final_answer_llm` from search params) with a strict JSON-array-of-queries prompt capped at 5; without an LLM it returns no gaps, and any parse/call failure degrades to "no gaps" with a warning — gap analysis never fails a run. It runs after EVERY synthesis (including the last) so the report can name what remains unresolved even when the iteration bound stops the loop.
- Iteration transitions stream as `iteration_started`/`iteration_complete` events (with iteration number, queries, and gap count) through the existing `update_run_progress` event channel. `report_v1.md` gains a `## Remaining gaps` section when the last analysis found any; `bundle.json` carries `iterations` and `remaining_gaps`.
- Budget interaction: each round's searches go through the task-16323 ledger per query (fan-out clamp + reserve before spend, settle after), so an iteration that cannot reserve stops cleanly through the existing `research_limit_exceeded` → `fail_run` path with partial artifacts. Collection artifacts (`plan.json` with accumulated sub-questions/iteration count, `collection_summary.json` per round) save before enforcement.
- Verified TDD: 5 new tests written first and watched failing (single-pass default, iterate-until-resolved with merged-evidence and event assertions, hard max-iterations bound with remaining-gaps reporting, budget-exhaustion mid-iteration, default gap_fn without LLM); full `Tests/Research/` = 75 passed; ruff clean. Files: `local_research_engine.py`, `test_local_research_engine.py`.
<!-- SECTION:NOTES:END -->
