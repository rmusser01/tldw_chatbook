---
id: TASK-34434
title: Fix flashrank rerank passages missing id so rerank actually ranks
status: Done
created_date: 2026-10-08 00:35
updated_date: 2026-10-09 05:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34413: rerank_results builds passages without id and validates ranked.index against real flashrank dicts (which return input dicts with score added) - AttributeError is swallowed and every real rerank silently falls back to original unranked order. The feature has likely never worked. Fix the validation/passages to the real flashrank contract with a live-path test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real flashrank output re-orders passages,Golden + fake-factory tests updated to the real contract,Fallback semantics preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 1 parked finding + task-1-report.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified flashrank 0.2.10 contract from installed package source (Ranker.py): rerank() mutates input passage dicts in place (adds 'score'), sorts list desc, returns same list; 'id' passes through untouched. Fix: passages now carry unique 'id' (results index); new _extract_flashrank_rank maps ranked items back by dict 'id', object-identity fallback for id-less input dicts, and tolerates attribute-style .index/.score; explicit len(ranked)==len(results) check; validate ALL entries then sort by reranker score desc before mutating (top_k applied after sort). Provenance contract preserved: plan-then-mutate, prior RRF/semantic scores+markers untouched on any fallback. TDD: new Tests/RAG_Search/test_flashrank_rerank_contract.py was red pre-fix (4 failures proving the swallowed AttributeError) -> 12/12 green post-fix incl. REAL-library smoke (cached weights, paris=0.9997 vs soup=0.0, flipped order). T1 fakes updated to real dict contract (test_flashrank_ranker_reuse.py inverts scores so broken id-mapping fails; test_local_citation_capture.py 4 call sites now dicts). Suites: reuse+citation+contract 122 passed; Tests/RAG_Search -k 'pipeline or rerank' 52 failed all pre-existing (RecoveryRequired + LLM-reranker/prompts-DB env, zero coupling to changed files); scope/middleware/notes trio failures pre-existing per T1 baseline. Files: tldw_chatbook/RAG_Search/pipeline_functions_simple.py, Tests/RAG_Search/test_flashrank_rerank_contract.py (new), Tests/RAG_Search/test_flashrank_ranker_reuse.py, Tests/RAG/test_local_citation_capture.py. ADR: none required (bugfix preserving existing boundaries).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
