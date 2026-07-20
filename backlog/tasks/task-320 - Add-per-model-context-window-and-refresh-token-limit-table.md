---
id: TASK-320
title: Add per-model context_window and refresh the stale token-limit table
status: To Do
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
- [ ] #1 A per-model `context_window` is resolvable (pattern-based, like the vision capability) for the providers/models the app supports
- [ ] #2 The hardcoded token-limit table reflects current model windows, or is derived from the capabilities config
- [ ] #3 Token-usage percentage/threshold logic (`token_counter.py:330` and its callers) is computed against the model input context window, not the output-token budget
- [ ] #4 Unit tests cover limit lookup for at least one current model per major provider
<!-- AC:END -->
