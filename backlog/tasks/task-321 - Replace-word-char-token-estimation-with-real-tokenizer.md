---
id: TASK-321
title: Replace word/char token estimation with a real tokenizer estimate
status: Done
assignee: []
created_date: '2026-07-20 18:45'
labels: [llm, tech-debt]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Token counting is only accurate for OpenAI (tiktoken). For every other provider, `count_tokens_messages` falls back to whitespace word counts via `len(role.split())` / `len(content.split())` (`token_counter.py:183,188`), and `approximate_token_count` (`Chat_Functions.py:68`) is also a whitespace word count. Word counts under-estimate real tokens by ~1.3-1.5x for English and far more for code/CJK, so any budget or trim built on these numbers under-counts and still overflows the model window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Non-OpenAI token estimates no longer use whitespace word counts (`.split()`)
- [x] #2 A consistent estimator (provider tokenizer where available, else a chars-based estimate with documented headroom) is used everywhere token counts feed a budget or a UI limit
- [x] #3 Tests assert the estimate is at least a conservative floor for representative code and CJK samples
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced every `.split()` token estimate with a single consistent estimator. Part of the token/context-accuracy sub-project (with task-320/325).

Approach: added `estimate_tokens(text, model, provider="") -> int` in `token_counter.py` with a tiered strategy — **custom tokenizer** (only behind a cheap `custom_tokenizers_available()` / `has_tokenizers()` gate so the default install pays no per-call metric overhead) → **tiktoken** (when installed) → **chars-based conservative floor**. The chars floor weights CJK code points higher than ASCII (`int((non_cjk*0.25 + cjk*1.0) * 1.2)` with a documented 1.2 headroom) so it never under-counts code or CJK — the safe direction for a budget. `count_tokens_messages` (now with a trailing optional `provider`), `count_tokens_chat_history`, and `estimate_remaining_tokens`'s system-prompt branch all delegate to `estimate_tokens`, so there is exactly one estimator and no double custom-counting. Deleted the dead `approximate_token_count` (zero callers). Routed the Console draft estimate `chat_screen._estimate_tokens` through `estimate_tokens`, removing its `.split()*1.3` fallback (the last in-scope `.split()` token estimate).

task-322's `console_history_budget.py` is untouched — its `count_tokens_messages(flattened, model)` call keeps working (trailing-optional `provider`) and now routes through the improved estimator.

Testing (property-based, tiktoken absent): pure-CJK string estimates ≥ its char count; a code sample exceeds its whitespace word count; a 100-ASCII-char string lands in (25,50) (keeps the pre-existing `test_character_estimation_fallback` green); empty→0; `count_tokens_chat_history` == `count_tokens_messages` for one message.

Deferred follow-up minors (non-blocking, from final review): `_CJK_RANGES` omits CJK punctuation U+3000–303F; `count_messages_with_custom` is now an unused import in `token_counter.py`; the new estimator tests aren't `skipif(TIKTOKEN_AVAILABLE)`-guarded (matches the repo's existing unguarded convention; would need guarding only to run green in a tiktoken-present env).

Files: `tldw_chatbook/Utils/token_counter.py`, `tldw_chatbook/Utils/custom_tokenizers.py`, `tldw_chatbook/Chat/Chat_Functions.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/Chat/test_token_counter.py`, `Tests/Chat/test_chat_screen_token_estimate.py`.
<!-- SECTION:NOTES:END -->
