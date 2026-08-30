---
id: TASK-21515
title: Briefing generation fails on reasoning-typed default models
status: Done
assignee: []
created_date: '2026-08-30 05:59'
updated_date: '2026-08-30 06:52'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of TASK-21513 (Daily Reports demo) reproduced: with the config-default deepseek-v4-flash, the ~15k-char briefing prompt makes reasoning consume all of BRIEFING_MAX_TOKENS=2000 in Subscriptions/briefing_service.py, so the provider returns finish=length with empty content and the briefing row fails with 'returned an empty response'. deepseek-chat completes fine. This is pre-existing Watchlists behavior, not a TASK-21513 regression, but it breaks the demo's one-click promise for users whose default endpoint is a reasoning-typed model. Candidate fixes: raise BRIEFING_MAX_TOKENS, exclude/override reasoning models for briefings, or provider-aware max_tokens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `deepseek_model_thinks_by_default(model)` predicate in `model_capabilities.py` recognizes the reasoning-typed DeepSeek families (`deepseek-v4-flash`, `deepseek-v4-pro`, `deepseek-reasoner`) and rejects `deepseek-chat`, lookalikes, and non-string input.
- [x] #2 Briefing generation against the native `deepseek` endpoint with a reasoning-typed model sends a widened completion budget (12000) instead of `BRIEFING_MAX_TOKENS` (2000), and every other endpoint/model combination keeps its plain budget.
- [x] #3 The cast call (`briefing_cast.py`) and the Library RAG answer call (`library_rag_answer_service.py`) apply the same reasoning-aware budget (12000 / 6000 respectively) with the same endpoint+model gate.
- [x] #4 All three `_effective_max_tokens` helpers treat endpoint casing/whitespace case-insensitively and unit-testably; targeted test files green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the per-family predicate `deepseek_model_thinks_by_default` to `model_capabilities.py`, following the established moonshot/zai per-family predicate idiom (compiled boundary-safe regex, `model: object` coercion, Google-style docstring).
2. Write RED tests first: predicate family-boundary tests in `Tests/LLM_Calls/test_chat_model_capability_predicates.py`; reasoning/non-reasoning budget assertions through each service's `_FakeChat` seam plus direct `_effective_max_tokens` unit tests in the three service test files.
3. Add `BRIEFING_REASONING_MAX_TOKENS = 12000` + `_effective_max_tokens(endpoint, model)` to `briefing_service.py` and wire it into `_invoke_chat` (the only place the budget is set).
4. Mirror in `briefing_cast.py` (`CAST_REASONING_MAX_TOKENS = 12000`) and `library_rag_answer_service.py` (`ANSWER_REASONING_MAX_TOKENS = 6000`, proportionate to that feature's deliberately short answer body), each with its own module-private helper per the copy-not-import idiom.
5. Run the four touched test files; verify GREEN; commit production + test + task files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reasoning-aware completion budgets shipped: deepseek_model_thinks_by_default predicate + _effective_max_tokens in briefing_service (12000), briefing_cast (12000), library_rag_answer_service (6000); TDD red->green; targeted tests green (2 pre-existing cast app-default failures unrelated, verified via stash).
<!-- SECTION:NOTES:END -->
