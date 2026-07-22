---
id: TASK-450
title: 'RAG reranker: remove unreachable literal fallback in _call_llm_impl'
status: To Do
assignee: []
created_date: '2026-07-21 20:07'
labels:
  - internal-prompts
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In `tldw_chatbook/RAG_Search/reranker.py` (~line 168), the `... or "You are a search result relevance evaluator."` inline fallback arm is now unreachable: each reranker's `__init__` always populates `config.system_prompt` from the registry, so the guarded expression can never fall through to that literal. Remove the dead arm. Pure cleanup, no behavior change. Deferred minor from the P1 whole-branch review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unreachable `or "You are a search result relevance evaluator."` fallback is removed from reranker.py
- [ ] #2 Existing reranker tests (Tests/RAG/test_reranker_internal_prompts.py and reranker suites) remain green
- [ ] #3 No behavior change: rerankers still resolve their system prompt from the registry / caller config
<!-- AC:END -->
