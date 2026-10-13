---
id: TASK-34895
title: Route summarization through the chat provider dispatch
status: In Progress
created_date: 2026-10-12 17:52
labels:
- simplification
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-13 14:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Summarization carries a stale second copy of the provider layer: 19 per-provider summarize_with_* functions with a hand-rolled if/elif dispatch alias table per-provider credential resolution (from legacy config sections) and 12 hand-rolled SSE loops. A summarization call is just one chat completion; routing through chat_api_call deletes ~3.9k LOC and unlocks every engine preset for summarization.

Evidence (verified on dev df4985e): Files: LLM_Calls/Summarization_General_Lib.py (9 cloud functions ~1.9k LOC dispatched by _dispatch_to_api at :333 with _CHAT_DISPATCH_NAME_ALIASES at :318) and LLM_Calls/Local_Summarization_Lib.py (9 local functions ~2k LOC). The alias-map comment at Summarization_General_Lib.py:326-330 admits engine presets are unusable today. Transport was already shared via owned_json_post (TASK-32853); everything above transport is per-provider. Keep model_capabilities predicates per TASK-19021/19051. No ADR blocks this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Single summarize core builds the prompt and calls chat_api_call,Per-provider summarize_with_* functions plus the dispatch chain and alias table removed,Hand-rolled SSE loops in the summarization libs removed,Ingest-consumed error strings preserved or ingest updated alongside,Engine-preset providers (moonshot zai etc) become selectable for summarization,model_capabilities sampling-param gating preserved on the generic path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map every caller of the two summarization libs and the ingest-consumed error strings 2. Characterize current behavior per provider (params resolution streaming) 3. Build the shared summarize core over chat_api_call 4. Repoint the dispatch cut over providers one at a time 5. Delete per-provider functions + alias table + SSE loops 6. Targeted tests green + engine presets selectable for summarization
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TRANCHE 1 LANDED (not yet Done - full AC pending). Structural map + 24-failure environmental baseline captured. Built _summarize_via_chat_api core + _SUMMARIZER_CHAT_SPECS in Summarization_General_Lib (zero diagnostic statements by design - the diagnostic-privacy ledger pins module statement counts) and rewired _SUMMARIZER_DISPATCH for groq deepseek mistral openrouter (plus the mistralai alias) away from the per-provider HTTP copies. Behavior preserved per spec: legacy model sourcing incl fallbacks (llama3-70b-8192 deepseek-chat mistral-large-latest wizardlm-2-8x22b) prompt assembly order (mistral keeps its reversed cp-before-text + trailing space) per-provider default temps the shared legacy system default dict-with-summary short-circuit list input via extract_text_from_segments the exact in-band error literals (missing-key / empty-extraction / generic templates incl groq's doubled name and openrouter's lowercase) and keyless refusal BEFORE the chat dispatch (encrypted-key suite initially caught the handler-side config trip - this machines RecoveryRequired env - which drove the better design). Streaming: _chat_stream_text_chunks translates chat SSE wire frames to plain text chunks (skips unparseable frames like the legacy generators; in-stream error frames become one Error during streaming chunk; wrapped in recovery_review.unqualified). New suite Tests/LLM_Calls/test_summarization_chat_dispatch.py (11 tests) pins endpoint/model/messages/system/temp/streaming forwarding + all string contracts + dispatch routing incl mistralai. Verification: full 13-file pinning suite - ZERO new failures vs baseline (24 pre-existing env failures unchanged) incl the diagnostic ledger and encrypted-key suites. REMAINING (tranche 2+): openai (max_completion_tokens swap lives in chat path already) anthropic (sampling gating in chat path; redirect-refusal pinned in BOTH modules - summarizer-side pin retarget needed) cohere (v2 wire pins at Session.post move to the chat handler equivalents) google huggingface (inputs-API to chat-handler wire change - behavior delta to document) the local set (llama kobold ooba tabby vllm ollama custom x2 local-llm - config-resolution pins live in test_llama_summarizer_config + test_kobold_tabby_config) then delete the 19 per-provider functions + alias-table remnants with the diagnostic-privacy ledger re-freeze (task-3796 correction precedent) and the privacy drivers retargeted to the core. Engine presets (moonshot zai aphrodite mlx) become selectable once specs extend - one line each.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
