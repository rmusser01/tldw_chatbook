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
updated_date: 2026-10-14 14:32
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
TRANCHES 1-3 + REVIEW ROUND 1 COMPLETE (deletion tranche pending; task stays In Progress). All 22 providers route through _summarize_via_chat_api: 9 cloud + 9 local + 4 NEW engine presets (moonshot zai aphrodite mlx_lm - selectable for summarization for the first time; require_key=False so the chat handler resolves modern api_settings keys). Review round 1 (requesting-code-review subagent in Qodo's place) verdict was No-with-fixes; ALL findings fixed: (1) engine presets were unreachable dead code with wrong legacy sections -> dispatch rows added + handler-side key resolution; (2) custom-openai endpoints' prompt order restored to the legacy text-first order (the template line was dead code in legacy); (3) handler-side ChatConfigurationError no longer masquerades as missing-key - only pre-dispatch key resolution produces the missing-key literal; (4) openrouter fallback model restored to the legacy mistralai/mistral-7b-instruct; (5) config-temperature sourcing restored for llama/ooba/vllm/ollama/custom x2 via spec.config_temp; (6) openrouter/huggingface keyless literals made marker-matching (legacy matched; groq-style literals never did - parity kept); (7) plain empty completions return empty string (openai-family legacy behavior - downstream falsy fallbacks fire) with the reasoning-only diagnostic kept for evidence-bearing empties (ported from llama for ALL providers); (8) explicit empty-string system messages preserved. New pins: 22-way dispatch-routing test (every spec key has a dispatch row - the unreachable-preset class can't recur) and a 22-way failure-marker contract test encoding legacy-matching parity (openai/openrouter/huggingface must match; customs' legacy phrasing exempt). Suite: 76 core tests + full 13-file pinning suite at ZERO new failures vs the 24-failure environmental baseline. REMAINING (tranche 4): delete the 19 per-provider functions + dead helpers, retarget the direct-drive tests (capabilities/redirect/cohere-wire/llama/kobold-tabby/defects), diagnostic-privacy ledger re-freeze (task-3796 correction precedent), request-timeouts AST retarget, internal-prompts call-site test swap for a core behavior pin. Deliberate minor deltas documented: ooba trailing-space glue, ollama model requirement now handler-side, google temperature now forwarded, shared system default for locals on direct calls.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
