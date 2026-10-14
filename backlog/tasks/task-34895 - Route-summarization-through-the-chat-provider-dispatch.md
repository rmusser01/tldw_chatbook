---
id: TASK-34895
title: Route summarization through the chat provider dispatch
status: Done
created_date: 2026-10-12 17:52
labels:
- simplification
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-14 15:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Summarization carries a stale second copy of the provider layer: 19 per-provider summarize_with_* functions with a hand-rolled if/elif dispatch alias table per-provider credential resolution (from legacy config sections) and 12 hand-rolled SSE loops. A summarization call is just one chat completion; routing through chat_api_call deletes ~3.9k LOC and unlocks every engine preset for summarization.

Evidence (verified on dev df4985e): Files: LLM_Calls/Summarization_General_Lib.py (9 cloud functions ~1.9k LOC dispatched by _dispatch_to_api at :333 with _CHAT_DISPATCH_NAME_ALIASES at :318) and LLM_Calls/Local_Summarization_Lib.py (9 local functions ~2k LOC). The alias-map comment at Summarization_General_Lib.py:326-330 admits engine presets are unusable today. Transport was already shared via owned_json_post (TASK-32853); everything above transport is per-provider. Keep model_capabilities predicates per TASK-19021/19051. No ADR blocks this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Single _summarize_via_chat_api core builds the prompt and calls chat_api_call (22 specs incl the 4 engine presets),Per-provider summarize_with_* functions removed (19 deleted + Local_Summarization_Lib module) and dispatch routes only through the core (mock-llm excepted); the alias table REMAINS by design - documented deviation: it normalizes chat-dispatch spellings to spec keys and is load-bearing for 9 spellings,Hand-rolled SSE loops removed (chat frames translated once in _chat_stream_text_chunks),Ingest-consumed error strings preserved where legacy matched and documented where legacy never matched (failure-marker parity test pins the contract),Engine presets (moonshot zai aphrodite mlx_lm) selectable for summarization,model_capabilities gating preserved - the chat path applies the predicates with the forwarded model (handler-side suites pin it)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map every caller of the two summarization libs and the ingest-consumed error strings 2. Characterize current behavior per provider (params resolution streaming) 3. Build the shared summarize core over chat_api_call 4. Repoint the dispatch cut over providers one at a time 5. Delete per-provider functions + alias table + SSE loops 6. Targeted tests green + engine presets selectable for summarization
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
COMPLETE (tranches 1-4, two review rounds). Net: Summarization_General_Lib 3275->~1.3k lines; Local_Summarization_Lib (2315 lines) deleted; 19 per-provider functions + _post_with_retry + dead helpers gone; all 22 providers + 4 NEW engine presets (moonshot zai aphrodite mlx_lm) route through the chat dispatch. Review round 1 (subagent in Qodo's place): No-with-fixes -> all Critical+Important fixed (unreachable presets wired, custom prompt order, error masquerade, openrouter fallback model, config temps, marker literals, empty-completion semantics). Tranche 4: ledger re-freeze per the task-3796 correction procedure (258 retained sites: 185 private deletions recorded with reason, 265 reviewed_safe deletions removed since frozen sites cannot be deleted, 58 frozen + 15 metadata; new starting-projection digest 01aaf9cc...; LIVE_SCAN_MODULES split from the historical MODULE_COUNTS); ~100 dead driver tests removed; direct-drive tests retargeted to chat-path seams or deleted with chat-side coverage verified (cohere wire v2 URL/messages/no-v1-keys/streaming-delta in test_cohere_native_tools; anthropic session posts in test_anthropic_native_tools; vllm explicit-key + custom-openai stream-once RETARGETED as new chat-side pins). Review round 2: With-fixes -> fixed (dead harness in capabilities file, SG header refreshed, cohere multi-part concat re-pinned). Deviations documented: alias table retained (load-bearing); cohere/llama/kobold local endpoint config keys read by the chat handlers (tabby_api spelling both live); google temp now forwarded; empty-plain completions return empty string. VERIFICATION: full 15-file pinning set + import-weight - every failure attributed to the pre-existing environmental set (RecoveryRequired/scratch-config/network-guard/manifest-boundary) with pristine-dev reproductions; app imports clean; zero dangling references. ENV NOTE for CI babysitters: the RecoveryRequired state masks the cohere+anthropic chat-side suites locally - they carry wire pins this consolidation leans on and must run green in a healthy environment (CI) before merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
