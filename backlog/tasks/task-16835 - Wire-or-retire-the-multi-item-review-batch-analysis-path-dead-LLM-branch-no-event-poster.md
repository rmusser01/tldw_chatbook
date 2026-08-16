---
id: TASK-16835
title: 'Wire or retire the multi-item review batch-analysis path (dead LLM branch, no event poster)'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - dead-code
  - event-handlers
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-16194 (PR #1671) repaired `Event_Handlers/multi_item_review_events.py`'s four
nonexistent `app.run_in_thread` calls, but its review surfaced a pre-existing gap it
correctly left unfixed: **`app.llm_api_client` is never assigned anywhere in the live
app**. `grep -rn "llm_api_client" tldw_chatbook/` finds only the guards that read it
(`multi_item_review_events.py:85`, `:174`) and the call through it (`:194`
`app.llm_api_client.chat_with_model`), and `git log --all -S "llm_api_client"` shows the
attribute was never introduced in `app.py` at any point in history (review16194 §5). The
`hasattr` guard has therefore always been False in production: every "LLM analysis"
silently falls back to `generate_placeholder_analysis`.

Verified still true at dev `ee741cf10` — and the situation is one step worse than the
review recorded: the only production consumer of this module is `app.py:11727-11730`,
which dispatches `handle_batch_analysis_start` on a `BatchAnalysisStartEvent`, and
**nothing constructs or posts that event anywhere in `tldw_chatbook/`** — its poster was
`MultiItemReviewWindow`, deleted as dead code by TASK-1010 (PR #1019). So the whole
batch-analysis handler path is unreachable, and even if reached it would only produce
placeholders.

Decide: either wire the feature (assign a real LLM client — the codebase's actual
dispatcher is the sync `chat_api_call()` in `Chat/Chat_Functions.py:789`, so the existing
`asyncio.to_thread` hop at `:194` is the right shape — and give the event a real poster),
or retire the module the way TASK-16196 retired the legacy Study handlers. Do not leave a
third state where the code looks maintained but cannot execute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An explicit wire-or-retire decision is recorded (owner call if product-facing)
- [ ] #2 If wired: `BatchAnalysisStartEvent` has a real production poster, `app.llm_api_client` (or a replacement seam) is genuinely assigned, and a test proves a batch analysis reaches a real (mockable) LLM dispatch instead of the placeholder
- [ ] #3 If retired: the module, its app.py dispatch branch, and its tests are removed with the same per-symbol reachability evidence TASK-16196 used
- [ ] #4 No silent placeholder fallback remains presented as an "LLM analysis" either way
<!-- AC:END -->
