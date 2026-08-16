---
id: TASK-3502
title: >-
  Reranker follow-ups: provider/model selection, cost surface, and two re-review
  residuals
status: In Progress
assignee: []
created_date: '2026-08-07 20:36'
updated_date: '2026-08-16 22:03'
labels:
  - rag
  - settings
dependencies:
  - TASK-3170
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-3170's Task 4 fixed the reranker factory so a reranking-enabled profile actually constructs and runs a reranker (it had never worked before -- a double-strategy TypeError meant reranking silently never activated in production). That fix surfaced follow-on gaps that were explicitly left out of Task 4's scope: Settings ▸ RAG's 'Enable reranking' toggle creates a bare RerankingConfig that defaults to provider=openai, model=gpt-3.5-turbo with no way to pick a different provider/model and no cost disclosure before enabling it, even though a single search can now issue up to 15 provider calls for LLM-driven reranking (pointwise scores each candidate individually). Two smaller, already-diagnosed residuals from Task 4's re-review rounds are folded in here rather than filed separately: (a) the reranking_degraded disclosure tag's cache-safety fix (copy-not-mutate) has no dedicated test coverage for the Pairwise/Listwise strategies, where the copy semantics actually matter differently than Pointwise; (b) BaseReranker's last_rerank_failures/last_rerank_total counters are instance state on a shared singleton reranker and are racy under concurrent search() calls -- diagnostic-only corruption (the disclosed count could be wrong), not a correctness bug in the reranked results themselves.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings ▸ RAG's Reranking fold lets the user choose the reranking provider and model, not just a bare on/off toggle defaulting to openai/gpt-3.5-turbo
- [ ] #2 Enabling reranking discloses, before the user commits, that reranking issues one provider call per candidate result (up to Rerank results many) and therefore has a real per-search cost
- [ ] #3 A regression test drives the real PairwiseReranker and ListwiseReranker strategies through the reranking_degraded copy-not-mutate path and confirms neither poisons a cached SearchResult
- [ ] #4 BaseReranker's per-call failure counters are safe under concurrent search() calls on the shared reranker singleton, or the diagnostic disclosure is scoped so a race cannot misattribute one search's failures to another's disclosed tag
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-16-reranker-followups-design.md
Plan: Docs/superpowers/plans/2026-08-16-reranker-followups.md
T1 engine honesty (AC#3, AC#4, note-b) -> T2 UI honesty (AC#1, AC#2, note-a) -> T3 closure + cross_encoder filing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Final whole-branch review (2026-08-07) surfaced two additional scope items to fold in here: (a) the reranking_skipped/reranking_degraded disclosure tags currently have ZERO UI consumers -- disclosure is metadata/log-only, so a Hybrid Full user with a dead reranker credential sees normal-looking results with no indication reranking silently skipped or degraded; (b) partial pointwise failure stamps rerank_score = original_score on failed rows, so a 14/15-failed rerank still renders " | reranked" on rows that were never actually rescored -- conservative in direction (no fabricated score), but an over-claim about what happened to those rows.
<!-- SECTION:NOTES:END -->
