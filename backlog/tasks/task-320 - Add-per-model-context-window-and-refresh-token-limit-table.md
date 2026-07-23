---
id: TASK-320
title: Add per-model context_window and refresh the stale token-limit table
status: Done
assignee: []
created_date: '2026-07-20 18:45'
labels: [llm, tech-debt]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no authoritative per-model context window anywhere in the codebase, which blocks correct token-based history bounding and makes the token gauge misleading. `model_capabilities.py` tracks only `vision`/`max_images` (no `context_window` field). `Utils/token_counter.py:82-111` (`MODEL_TOKEN_LIMITS`) predates current models (no gpt-4o, claude-3.5/4, gemini-1.5/2, o1/o3), so `get_model_token_limit` (`token_counter.py:260`) falls back to wrong conservative defaults (e.g. Anthropic 100k vs real 200k). Separately, the token-usage percentage is computed against the wrong ceiling: `chat_token_events.py:149` passes the ~2048-token output-response budget as the limit rather than the model's input window (`token_counter.py:330`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A per-model `context_window` is resolvable (pattern-based, like the vision capability) for the providers/models the app supports
- [x] #2 The hardcoded token-limit table reflects current model windows, or is derived from the capabilities config
- [x] #3 Token-usage percentage/threshold logic (`token_counter.py:330` and its callers) is computed against the model input context window, not the output-token budget
- [x] #4 Unit tests cover limit lookup for at least one current model per major provider
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the model context window authoritative and config-overridable, and fixed the token gauge's denominator. Part of the token/context-accuracy sub-project (with task-321/325); feeds the task-322 Console history trimmer through the shared `token_counter` seam with no change to 322's code.

Approach:
- **`model_capabilities.py`**: added a per-model `context_window` capability parallel to `vision`/`max_images` — on the direct mappings (`DEFAULT_MODEL_CAPABILITIES`) and on anchored OpenAI/Anthropic family patterns (`DEFAULT_MODEL_PATTERNS`); Google windows resolve via direct mappings + table since its `(pro|flash)` pattern spans non-uniform windows. Added `get_context_window(provider, model)` (method + module fn). Fixed a **latent case-sensitivity bug**: the provider→pattern lookup was case-sensitive while callers pass mixed/lowercase — now case-insensitive via a lower-cased provider index (hardens the existing `is_vision_capable` path too).
- **`token_counter.py`**: rewrote `get_model_token_limit` to resolve capability `context_window` FIRST, then an exact table match, then the **longest** table prefix (fixes `gpt-4` shadowing `gpt-4-turbo`), then a conservative provider default. Refreshed `MODEL_TOKEN_LIMITS` with current models (gpt-4o/4.1, o1/o3/o4, claude-3.5/3.7/4, gemini-1.5/2, mistral-large) while preserving every test-pinned value; the only value correction is `gpt-3.5-turbo` 4096→16385. Provider defaults stay conservative (only `anthropic` bumped 100000→200000, the safe floor for modern Claude) because over-estimating a window is the only way to overflow the 322 budget on dispatch.
- **`chat_token_events.py`** (AC#3): both gauge sites now divide usage by the model **input** window (`total_limit`) via a small `_resolve_token_display_limit(total_limit, custom_limit)` helper, not the ~2048-token output reservation; the manual custom-limit override is preserved.

Testing: capability resolution incl. case-insensitivity + anchored-pattern/table agreement; `get_model_token_limit` per major provider + capabilities-first (monkeypatched sentinel) + discriminating longest-prefix (`gpt-4-32k-custom`→32768); gauge helper (input-window default / custom override). tiktoken is absent in the venv so estimator tests target the chars path.

Files: `tldw_chatbook/model_capabilities.py`, `tldw_chatbook/Utils/token_counter.py`, `tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py`, `Tests/test_model_capabilities.py`, `Tests/Chat/test_token_counter.py`, `Tests/Chat/test_token_display_limit.py`.
<!-- SECTION:NOTES:END -->
