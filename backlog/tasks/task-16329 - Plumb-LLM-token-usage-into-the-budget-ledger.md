---
id: TASK-16329
title: Plumb LLM token usage into the budget ledger
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 16:46'
updated_date: '2026-08-15 17:38'
labels:
  - research
  - budget
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The budget ledger (task-16323) has token reserve/settle infrastructure but no usage signal: chat_api_call returns text only and provider handlers discard usage, so max_tokens cannot be enforced honestly. Add a context-scoped usage recorder at the chat_api_call seam (estimate-based accounting now, real-usage capable when providers expose it) and settle it into the ledger so runs with max_tokens are enforced between LLM-bearing units of work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A context-scoped usage recorder exists at the chat_api_call seam and records prompt plus completion token estimates for non-streaming string responses,chat_api_call only records when a recorder is active (zero overhead and zero behavior change otherwise),The engine settles recorded usage into the ledger after each LLM-bearing call and checks the token budget before the next one,Exhausted max_tokens stops the run cleanly through the existing research_limit_exceeded path,The ledger snapshot marks token counts as estimates,Tests cover the recorder scoping the chat_api_call recording the engine settlement and the enforcement boundary
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- New `Chat/usage_recorder.py`: context-scoped `UsageTokenRecorder` + `usage_scope()` (contextvar — concurrent runs keep separate ledgers and scopes survive await points). `estimate_tokens` (~4 chars/token, floored at 1) is the documented ESTIMATE source; `record_usage(prompt_tokens, completion_tokens)` is the exact-count path for when providers expose real usage — same recorder, no further changes needed then.
- `chat_api_call` records estimated prompt (serialized `messages_payload`) + completion tokens for non-streaming string responses WHEN a recorder is active — one contextvar get otherwise, wrapped so accounting can never break a call. This is the plumbing at the one call path the deep-search pipeline uses; per-provider real-usage extraction remains the future upgrade (handlers currently return text only).
- Engine: every LLM-bearing call (`search_fn` sub-query generation, `analyze_fn` synthesis, `gap_fn`, follow-up answerer) runs through `_llm_bounded_call` — token budget checked BEFORE the call (post-settlement enforcement, since estimates arrive after calls complete), usage settled AFTER into the ledger, which persists at packaging time (after the last settlement). Exhaustion stops the run through the existing `research_limit_exceeded` → `fail_run` path with partial artifacts. `BudgetLedger.check_tokens()` + `tokens_estimated: True` snapshot flag added.
- Verified TDD: 7 recorder/chat-hook tests (scope isolation, await survival, recording with a faked `API_CALL_HANDLERS` entry, no-recorder zero-change) + 2 ledger tests + 2 engine tests (settlement into the persisted artifact; max_tokens refusal between synthesis and gap analysis) — all written first and watched failing; `Tests/Research/ + Tests/Chat/test_usage_recorder.py + test_anthropic_prefix_stability.py` = 109 passed; ruff clean on all touched files.
<!-- SECTION:NOTES:END -->
